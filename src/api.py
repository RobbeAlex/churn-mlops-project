import os
import joblib
import pandas as pd
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict

# ==========================================
# CONFIGURACIÓN DE ENTORNO Y RUTAS
# ==========================================

raiz_proyecto = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
    config = yaml.safe_load(f)

model = None

# ==========================================
# GESTIÓN DEL CICLO DE VIDA (LIFESPAN)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
        config_l = yaml.safe_load(f)

    ruta_m = os.path.join(raiz_proyecto, config_l['paths']['model_path'].replace('/', os.sep))
    if os.path.exists(ruta_m):
        model = joblib.load(ruta_m)
    yield
    model = None

app = FastAPI(title="Churn Prediction API - UDG", lifespan=lifespan)

# Carga inicial preventiva del modelo
ruta_modelo = os.path.join(raiz_proyecto, config['paths']['model_path'].replace('/', os.sep))
try:
    if os.path.exists(ruta_modelo):
        model = joblib.load(ruta_modelo)
    else:
        model = None
except Exception as e:
    print(f"Error crítico al cargar el modelo de respaldo: {e}")
    model = None

# ==========================================
# ESQUEMAS DE VALIDACIÓN (PYDANTIC V2)
# ==========================================

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

# ==========================================
# ENDPOINTS / RUTAS DE LA API
# ==========================================

@app.post("/predict")
async def predict(payload: ChurnInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo predictivo no se encuentra cargado en el servidor."
        )
    try:
        # 1. Crear DataFrame inicial con los datos crudos del usuario
        df_input = pd.DataFrame([payload.model_dump(by_alias=True)])
        
        mapeo_binario = {
            "Male": 1, "Female": 0,
            "Yes": 1, "No": 0
        }
        
        # 2. ALINEACIÓN: Si el modelo tiene guardadas las columnas de entrenamiento
        if hasattr(model, "feature_names_in_"):
            columnas_entrenamiento = model.feature_names_in_
            df_final = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
            
            for col in df_input.columns:
                valor = df_input.iloc[0][col]
                if col in df_final.columns:
                    if isinstance(valor, str) and valor in mapeo_binario:
                        df_final[col] = mapeo_binario[valor]
                    else:
                        df_final[col] = valor
                else:
                    nombre_dummy = f"{col}_{valor}"
                    if nombre_dummy in df_final.columns:
                        df_final[nombre_dummy] = 1
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
        print(f"Error crítico en la predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Error al procesar la predicción: {str(e)}"
        )