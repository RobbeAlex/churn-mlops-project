import pytest
import yaml
import os
import sys
import pandas as pd
from fastapi.testclient import TestClient

# Mantenemos tu lógica de rutas, es robusta para entornos locales
ruta_actual = os.path.abspath(os.path.dirname(__file__))
raiz_proyecto = os.path.dirname(ruta_actual)
if raiz_proyecto not in sys.path:
    sys.path.append(raiz_proyecto)

from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model
from src.api import app  # Importamos la API para probar el endpoint

client = TestClient(app)


@pytest.fixture
def config():
    """Carga la configuración del archivo YAML."""
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')
    # Corregido: 'with open' en lugar de 'withopen'
    with open(ruta_config, 'r') as file:
        return yaml.safe_load(file)


def test_load_and_preprocess_data_integrity(config):
    """
    Verifica no solo que los datos existan, sino que la limpieza
    haya sido efectiva (sin nulos en columnas clave).
    """
    # Verificamos que el archivo de datos existe antes de intentar cargar
    if not os.path.exists(config['paths']['raw_data']):
        pytest.skip("Archivo de datos no encontrado en data/raw/")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    # Validaciones de estructura
    assert not X_train.empty, "El set de entrenamiento está vacío"
    assert X_train.shape[1] == X_test.shape[1], "Diferencia de columnas entre train y test"

    # Validación de limpieza (Rol: Data Engineer)
    assert X_train.isnull().sum().sum() == 0, "Se encontraron valores nulos tras el preprocesamiento"


def test_train_and_save_model_logic(config, tmp_path):
    """
    Prueba el entrenamiento pero usa una ruta temporal para no
    sobrescribir tu mejor modelo actual.
    """
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    # Reducimos datos para que el test sea rápido
    X_train_sub, y_train_sub = X_train.head(50), y_train.head(50)

    # Usamos una ruta temporal de pytest para el modelo
    temp_model_path = tmp_path / "test_model.pkl"
    config['paths']['model_path'] = str(temp_model_path)
    config['model']['name'] = 'LogisticRegression'

    metrics = train_and_save_model(X_train_sub, y_train_sub, X_test.head(10), y_test.head(10), config)

    assert 'accuracy' in metrics
    assert os.path.exists(temp_model_path), "El modelo no se guardó en la ruta especificada"


def test_api_predict_endpoint():
    """
    Prueba de integración de la API (Rol: QA & Production Engineer).
    """
    # Payload de ejemplo basado en tu README
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 5,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 400.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "probability" in json_data
