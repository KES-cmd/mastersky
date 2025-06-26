from catboost import CatBoostClassifier
from datetime import datetime
from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Request
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json

app = FastAPI()

# Настройка путей
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')),
          name='static')


@app.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    return FileResponse(str(BASE_DIR / 'static' / 'favicon.ico'))

# Папка для сохранения результатов
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Определение столбцов в соответствии с типами из модели
NUM_COLUMNS = [
    'Age', 'Cholesterol', 'Heart_Rate', 'Exercise_Hours_Per_Week', 'Diet',
    'Stress_Level', 'Sedentary_Hours_Per_Day', 'Income', 'BMI',
    'Triglycerides', 'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day',
    'Blood_Sugar', 'CK_MB', 'Troponin', 'Systolic_blood_pressure',
    'Diastolic_blood_pressure'
]

CAT_COLUMNS = [
    'Diabetes', 'Family_History', 'Smoking', 'Obesity',
    'Alcohol_Consumption', 'Previous_Heart_Problems',
    'Medication_Use', 'Gender'
]

# Поля, требующие специальной обработки
SPECIAL_FIELDS = {
    'Diet': 'int',
    'Exercise_Hours_Per_Week': 'int',
    'Gender': 'gender'
}

MODEL_PATH = BASE_DIR / 'catboost_model.cbm'
try:
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    print('CatBoost model loaded successfully')
    print('Model feature names:', model.feature_names_)
    print('Model cat features indices:', model.get_cat_feature_indices())
except Exception as e:
    raise RuntimeError(f'Failed to load CatBoost model: {str(e)}')


class PatientData:
    def __init__(self, **kwargs):
        self.data = {}
        for k, v in kwargs.items():
            if v is None or v == '':
                self.data[k] = 0
                continue
            if k in SPECIAL_FIELDS:
                if SPECIAL_FIELDS[k] == 'int':
                    self.data[k] = self._safe_int(v)
                elif SPECIAL_FIELDS[k] == 'gender':
                    self.data[k] = self._parse_gender(v)
                continue
            if k in NUM_COLUMNS:
                self.data[k] = self._safe_float(v)
            elif k in CAT_COLUMNS:
                self.data[k] = self._safe_int(v)

    def _safe_float(self, value):
        """Безопасное преобразование в float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(self, value):
        """Безопасное преобразование в int"""
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _parse_gender(self, value):
        """Преобразование пола в числовой формат"""
        try:
            value = str(value).lower()
            if value in ['1', 'female', 'жен', 'женский']:
                return 1  # female
            return 0  # male (по умолчанию)
        except (ValueError, TypeError):
            return 0

    def _prepare_features(self):
        """Подготавливает данные в порядке, ожидаемом моделью"""
        feature_map = {
            'Age': self.data['Age'] / 100.0,
            'Cholesterol': self.data['Cholesterol'],
            'Heart rate': self.data['Heart_Rate'],
            'Diabetes': self.data['Diabetes'],
            'Family History': self.data['Family_History'],
            'Smoking': self.data['Smoking'],
            'Obesity': self.data['Obesity'],
            'Alcohol Consumption': self.data['Alcohol_Consumption'],
            'Exercise Hours Per Week': self.data['Exercise_Hours_Per_Week'],
            'Diet': self.data['Diet'],
            'Previous Heart Problems': self.data['Previous_Heart_Problems'],
            'Medication Use': self.data['Medication_Use'],
            'Stress Level': self.data['Stress_Level'],
            'Sedentary Hours Per Day': self.data['Sedentary_Hours_Per_Day'],
            'Income': self.data['Income'],
            'BMI': self.data['BMI'],
            'Triglycerides': self.data['Triglycerides'],
            'Physical Activity Days Per Week':
            self.data['Physical_Activity_Days_Per_Week'],
            'Sleep Hours Per Day': self.data['Sleep_Hours_Per_Day'],
            'Blood sugar': self.data['Blood_Sugar'],
            'CK-MB': self.data['CK_MB'],
            'Troponin': self.data['Troponin'],
            'Gender': self.data['Gender'],
            'Systolic blood pressure': self.data['Systolic_blood_pressure'],
            'Diastolic blood pressure': self.data['Diastolic_blood_pressure']
        }

        # Получаем признаки в правильном порядке, как ожидает модель
        ordered_features = [feature_map[name] for name in model.feature_names_]
        # Для отладки
        print('Подготовленные признаки:', ordered_features)
        print('Типы признаков:', [type(x) for x in ordered_features])
        print('Соответствие имен:', list(zip(model.feature_names_,
                                         ordered_features)))
        return ordered_features

    def predict_risk(self):
        """Использует CatBoost модель для предсказания"""
        try:
            features = self._prepare_features()
            risk_percentage = model.predict_proba([features])[0][1] * 100
            return round(float(risk_percentage), 1)
        except Exception as e:
            raise Exception(f'Prediction failed: {str(e)}')


def save_prediction(data: dict, prediction: float):
    """Сохраняет результат предсказания в JSON файл"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'prediction_{timestamp}.json'
    filepath = RESULTS_DIR / filename
    result = {
        'input_data': data,
        'prediction': {
            'risk_percentage': prediction,
            'timestamp': datetime.now().isoformat()
        },
        'model_info': {
            'type': 'CatBoost',
            'version': model.get_params().get('model_version', 'unknown')
        }
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return filename


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict', response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: str = Form(...),
    Cholesterol: str = Form(...),
    Heart_Rate: str = Form(...),
    Exercise_Hours_Per_Week: str = Form(...),
    Diet: str = Form(...),
    Stress_Level: str = Form(...),
    Sedentary_Hours_Per_Day: str = Form(...),
    Income: str = Form(...),
    BMI: str = Form(...),
    Triglycerides: str = Form(...),
    Physical_Activity_Days_Per_Week: str = Form(...),
    Sleep_Hours_Per_Day: str = Form(...),
    Blood_Sugar: str = Form(...),
    CK_MB: str = Form(...),
    Troponin: str = Form(...),
    Systolic_blood_pressure: str = Form(...),
    Diastolic_blood_pressure: str = Form(...),
    Gender: str = Form(...),
    Diabetes: str = Form(...),
    Family_History: str = Form(...),
    Smoking: str = Form(...),
    Obesity: str = Form(...),
    Alcohol_Consumption: str = Form(...),
    Previous_Heart_Problems: str = Form(...),
    Medication_Use: str = Form(...),
):
    # Собираем все данные формы
    form_data = {
        'Age': Age,
        'Cholesterol': Cholesterol,
        'Heart_Rate': Heart_Rate,
        'Exercise_Hours_Per_Week': Exercise_Hours_Per_Week,
        'Diet': Diet,
        'Stress_Level': Stress_Level,
        'Sedentary_Hours_Per_Day': Sedentary_Hours_Per_Day,
        'Income': Income,
        'BMI': BMI,
        'Triglycerides': Triglycerides,
        'Physical_Activity_Days_Per_Week': Physical_Activity_Days_Per_Week,
        'Sleep_Hours_Per_Day': Sleep_Hours_Per_Day,
        'Blood_Sugar': Blood_Sugar,
        'CK_MB': CK_MB,
        'Troponin': Troponin,
        'Systolic_blood_pressure': Systolic_blood_pressure,
        'Diastolic_blood_pressure': Diastolic_blood_pressure,
        'Gender': Gender,
        'Diabetes': Diabetes,
        'Family_History': Family_History,
        'Smoking': Smoking,
        'Obesity': Obesity,
        'Alcohol_Consumption': Alcohol_Consumption,
        'Previous_Heart_Problems': Previous_Heart_Problems,
        'Medication_Use': Medication_Use,
    }

    try:
        patient = PatientData(**form_data)
        risk_percentage = patient.predict_risk()
        # Определяем уровень риска
        if risk_percentage < 30:
            risk_level = 'Low Risk'
            risk_class = 'low'
        elif risk_percentage < 70:
            risk_level = 'Medium Risk'
            risk_class = 'medium'
        else:
            risk_level = 'High Risk'
            risk_class = 'high'
        # Сохраняем результаты
        json_filename = save_prediction(form_data, risk_percentage)
        # Подготавливаем данные для шаблона
        result_data = {
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'risk_class': risk_class,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'json_filename': json_filename
        }
        return templates.TemplateResponse('index.html', {
            'request': request,
            'result': result_data,
            'form_data': form_data
        })
    except Exception as e:
        # Логируем ошибку
        print(f'Error during prediction: {str(e)}')
        return templates.TemplateResponse('index.html', {
            'request': request,
            'error': str(e)
        })


@app.get('/download/{filename}')
async def download_file(filename: str):
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(filepath, filename=f'heart_risk_report.json')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
