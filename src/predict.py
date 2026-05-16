import os
import joblib
import pandas as pd
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict

raiz_proyecto = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
    config = yaml.safe_load(f)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
        config_lifespan = yaml.safe_load(f)

    config_path = config_lifespan['paths']['model_path']
    if config_path.startswith('/app/'):
        config_path = config_path.replace('/app/', '', 1)
    elif config_path.startswith('/'):
        config_path = config_path.lstrip('/')
        
    ruta_modelo_lifespan = os.path.join(raiz_proyecto, config_path.replace('/', os.sep))
    model = joblib.load(ruta_modelo_lifespan)
    yield
    model = None

app = FastAPI(title="Churn Prediction API - UDG", lifespan=lifespan)

# Carga preventiva inicial
config_path_init = config['paths']['model_path']
if config_path_init.startswith('/app/'):
    config_path_init = config_path_init.replace('/app/', '', 1)
elif config_path_init.startswith('/'):
    config_path_init = config_path_init.lstrip('/')
ruta_modelo_init = os.path.join(raiz_proyecto, config_path_init.replace('/', os.sep))

try:
    if os.path.exists(ruta_modelo_init):
        model = joblib.load(ruta_modelo_init)
    else:
        print(f"Aviso: El archivo del modelo no se encontró en la ruta inicial: {ruta_modelo_init}")
except Exception as e:
    print(f"Error crítico al cargar el modelo de respaldo: {e}")
    model = None


class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: int
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    model_config = ConfigDict(populate_by_name=True)


@app.post("/predict")
async def predict(payload: ChurnInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo predictivo no se encuentra cargado en el servidor."
        )
    try:
        # 1. Crear DataFrame inicial con los datos crudos del JSON recibido
        df_input = pd.DataFrame([payload.model_dump(by_alias=True)])
        
        # 2. Aplicar exactamente las mismas transformaciones básicas del pipeline
        df_input['gender'] = df_input['gender'].map({'Female': 1, 'Male': 0})
        df_input['Partner'] = df_input['Partner'].map({'Yes': 1, 'No': 0})
        
        # Aplicar dummificación instantánea para variables categóricas de texto
        df_procesado = pd.get_dummies(df_input)
        
        # 3. Alineación directa usando el vector de características del entrenamiento
        if hasattr(model, "feature_names_in_"):
            columnas_entrenamiento = model.feature_names_in_
            df_final = df_procesado.reindex(columns=columnas_entrenamiento, fill_value=0)
        else:
            df_final = df_procesado

        # 4. Inferencia
        pred = model.predict(df_final)[0]
        prob = model.predict_proba(df_final)[0][1]

        return {
            "prediction": int(pred),
            "label": "Churn" if pred == 1 else "No Churn",
            "probability": round(float(prob), 2)
        }

    except Exception as e:
        print(f"Error crítico en la predicción de la API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Error al procesar la predicción: {str(e)}"
        )
