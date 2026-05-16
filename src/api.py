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

# Ruta raíz del proyecto (robusto dentro y fuera de Docker)
raiz_proyecto = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cargar configuración global
with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Variable global del modelo
model = None

# ==========================================
# GESTIÓN DEL CICLO DE VIDA (LIFESPAN)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    ruta_modelo = os.path.join(raiz_proyecto, config['paths']['model_path'].replace('/', os.sep))
    model = joblib.load(ruta_modelo)
    yield
    model = None

# Instanciar la aplicación FastAPI
app = FastAPI(title="Churn Prediction API - UDG", lifespan=lifespan)

# Carga inicial preventiva del modelo para scripts/tests que acceden directo al módulo
ruta_modelo = os.path.join(raiz_proyecto, config['paths']['model_path'].replace('/', os.sep))
try:
    model_data = joblib.load(ruta_modelo)
    model = model_data
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
    
    # Sintaxis oficial moderna para Pydantic v2
    model_config = ConfigDict(populate_by_name=True)

# ==========================================
# ENDPOINTS / RUTAS DE LA API
# ==========================================

@app.post("/predict")
async def predict(payload: ChurnInput):
    try:

        # 1. Convertir la entrada en DataFrame
        df_input = pd.DataFrame([payload.model_dump(by_alias=True)])

        # 2. Alinear columnas según entrenamiento (opcional si tienes feature_names_in_)
        if hasattr(model, "feature_names_in_"):
            columnas_entrenamiento = model.feature_names_in_
            df_final = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)

        # 1. Crear DataFrame inicial con los datos crudos del usuario
        df_input = pd.DataFrame([payload.model_dump(by_alias=True)])
        
        # Mapeo explícito para variables de texto binarias que no son dummies
        mapeo_binario = {
            "Male": 1, "Female": 0,
            "Yes": 1, "No": 0
        }
        
        # 2. ALINEACIÓN: Si el modelo tiene guardadas las columnas de entrenamiento
        if hasattr(model, "feature_names_in_"):
            columnas_entrenamiento = model.feature_names_in_
            df_final = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
            
            # Recorremos el payload para mapear tanto numéricas como categóricas

            for col in df_input.columns:
                valor = df_input.iloc[0][col]
                
                # Caso A: Es una variable que coincide directamente en nombre (como 'gender' o numéricas)
                if col in df_final.columns:
                    # Si el valor es un string ("Male", "Yes"), lo transformamos usando el mapa
                    if isinstance(valor, str) and valor in mapeo_binario:
                        df_final[col] = mapeo_binario[valor]
                    else:
                        df_final[col] = valor
                
                # Caso B: Es una variable categórica pura que se convirtió en Dummy ("Columna_Valor")
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