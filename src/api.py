import os
import joblib
import pandas as pd
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ruta raíz del proyecto (robusto dentro y fuera de Docker)
raiz_proyecto = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class ChurnInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int = Field(alias="MultipleLines_No phone service")
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int = Field(alias="InternetService_Fiber optic")
    InternetService_No: int
    OnlineSecurity_No_internet_service: int = Field(alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int = Field(alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int = Field(alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int = Field(alias="TechSupport_No internet service")
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int = Field(alias="StreamingTV_No internet service")
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int = Field(alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: int
    Contract_One_year: int = Field(alias="Contract_One year")
    Contract_Two_year: int = Field(alias="Contract_Two year")
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card_automatic: int = Field(alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(alias="PaymentMethod_Mailed check")

    model_config = {"populate_by_name": True}


@app.post("/predict")
async def predict(input_data: ChurnInput):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    try:
        df_input = pd.DataFrame([input_data.dict(by_alias=True)])

        if hasattr(model, "feature_names_in_"):
            columnas_entrenamiento = model.feature_names_in_
            df_final = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
            for col in df_input.columns:
                if col in df_final.columns:
                    df_final[col] = df_input[col]
        else:
            df_final = df_input

        pred = model.predict(df_final)[0]
        prob = model.predict_proba(df_final)[0][1]

        return {
            "prediction": int(pred),
            "label": "Churn" if pred == 1 else "No Churn",
            "probability": round(float(prob), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))