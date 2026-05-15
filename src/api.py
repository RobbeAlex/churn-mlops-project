import os
import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuración de rutas dinámicas para Zapopan/Windows
ruta_script = os.path.abspath(__file__)
raiz_proyecto = os.path.dirname(os.path.dirname(ruta_script))

app = FastAPI(title="Churn Prediction API - UDG")

# 1. Cargar configuración y modelo
with open(os.path.join(raiz_proyecto, 'config', 'params.yaml'), 'r') as f:
    config = yaml.safe_load(f)

ruta_modelo = os.path.join(raiz_proyecto, config['paths']['model_path'].replace('/', os.sep))

# Cargamos el modelo y extraemos las columnas que espera
try:
    model_data = joblib.load(ruta_modelo)
    # Si guardaste el modelo directamente, lo asignamos.
    # Si guardaste un dict con columnas, lo extraemos.
    model = model_data
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")
    model = None


class ChurnInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    Partner: int


@app.post("/predict")
async def predict(input_data: ChurnInput):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    try:
        # 2. Crear DataFrame con la entrada
        df_input = pd.DataFrame([input_data.dict()])

        # 3. ALINEACIÓN: El modelo espera ~30 columnas por los dummies
        # Recuperamos las columnas del entrenamiento.
        # Si el modelo es de scikit-learn, podemos obtenerlas de sus atributos si se guardaron
        if hasattr(model, "feature_names_in_"):
            columnas_entrenamiento = model.feature_names_in_
            # Creamos un DataFrame con ceros para todas las columnas faltantes
            df_final = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
            # Llenamos solo las que el usuario envió
            for col in df_input.columns:
                if col in df_final.columns:
                    df_final[col] = df_input[col]
        else:
            # Si no tenemos feature_names_in_, pasamos el DF tal cual (esto podría dar error)
            df_final = df_input

        # 4. Predicción
        pred = model.predict(df_final)[0]
        prob = model.predict_proba(df_final)[0][1]

        return {
            "prediction": int(pred),
            "label": "Churn" if pred == 1 else "No Churn",
            "probability": round(float(prob), 2)
        }

    except Exception as e:
        # Esto imprimirá el error real en tu terminal de PyCharm
        raise HTTPException(status_code=500, detail=str(e))