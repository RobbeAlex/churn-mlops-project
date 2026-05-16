import os
import joblib
import pandas as pd
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, ConfigDict

# Ruta raíz del proyecto (robusto dentro y fuera de Docker)
raiz_proyecto = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cargar configuración global
with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Variable global del modelo
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    ruta_modelo = os.path.join(raiz_proyecto, config['paths']['model_path'].replace('/', os.sep))
    model = joblib.load(ruta_modelo)
    yield
    model = None

app = FastAPI(title="Churn Prediction API - UDG", lifespan=lifespan)

ruta_modelo = os.path.join(raiz_proyecto, config['paths']['model_path'].replace('/', os.sep))

# Cargamos el modelo y extraemos las columnas que espera
try:
    model_data = joblib.load(ruta_modelo)
    model = model_data
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")
    model = None

# Definimos el esquema mapeando los nombres reales con espacios mediante alias
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
   
    # Sintaxis oficial moderna para Pydantic v2
    model_config = ConfigDict(populate_by_name=True)

@app.post("/predict")
async def predict(payload: ChurnInput):
    try:
        # 1. Convertir la entrada en DataFrame
        df_input = pd.DataFrame([payload.model_dump(by_alias=True)])

        # 2. Alinear columnas según entrenamiento (opcional si tienes feature_names_in_)
        if hasattr(model, "feature_names_in_"):
            columnas_entrenamiento = model.feature_names_in_
            df_final = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
            for col in df_input.columns:
                if col in df_final.columns:
                    df_final[col] = df_input[col]
        else:
            df_final = df_input

        # 3. Realizar la predicción
        pred = model.predict(df_final)[0]
        prob = model.predict_proba(df_final)[0][1]

        return {
            "prediction": int(pred),
            "label": "Churn" if pred == 1 else "No Churn",
            "probability": round(float(prob), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
