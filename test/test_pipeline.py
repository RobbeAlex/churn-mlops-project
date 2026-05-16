import pytest
import yaml
import os
import sys
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

ruta_actual = os.path.abspath(os.path.dirname(__file__))
raiz_proyecto = os.path.dirname(ruta_actual)
if raiz_proyecto not in sys.path:
    sys.path.append(raiz_proyecto)

from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model
import src.api as api_module
from src.api import app as api_app
from src.predict import app as predict_app
from src.main import main as main_entrypoint

client_api = TestClient(api_app)
client_predict = TestClient(predict_app)


@pytest.fixture
def config():
    """Carga la configuración del archivo YAML."""
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')
    with open(ruta_config, 'r') as file:
        return yaml.safe_load(file)


@pytest.fixture
def valid_payload():
    return {
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


def test_load_and_preprocess_data_integrity(config):
    if not os.path.exists(config['paths']['raw_data']):
        pytest.skip("Archivo de datos no encontrado")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    assert not X_train.empty
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.isnull().sum().sum() == 0


def test_train_and_save_model_logic(config):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    # REPARACIÓN AQUÍ: Filtramos para garantizar que entren ejemplos de ambas clases (0 y 1)
    df_train_full = pd.concat([X_train, y_train], axis=1)
    df_test_full = pd.concat([X_test, y_test], axis=1)
    
    # Tomamos muestras balanceadas para el entorno de pruebas rápido
    train_class_0 = df_train_full[df_train_full['Churn'] == 0].head(10)
    train_class_1 = df_train_full[df_train_full['Churn'] == 1].head(10)
    train_df = pd.concat([train_class_0, train_class_1], axis=0).sample(frac=1, random_state=42)
    
    test_class_0 = df_test_full[df_test_full['Churn'] == 0].head(5)
    test_class_1 = df_test_full[df_test_full['Churn'] == 1].head(5)
    test_df = pd.concat([test_class_0, test_class_1], axis=0).sample(frac=1, random_state=42)

    real_processed_dir = os.path.join(raiz_proyecto, 'data', 'processed')
    os.makedirs(real_processed_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(real_processed_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(real_processed_dir, 'test.csv'), index=False)

    # Cobertura de caminos lógicos en model_trainer (Logistic Regression)
    config['model']['name'] = 'LogisticRegression'
    config['paths']['model_path'] = "models/test_lr.pkl"
    metrics_lr = train_and_save_model(config)
    assert 'accuracy' in metrics_lr

    # Cobertura de caminos lógicos en model_trainer (Random Forest)
    config['model']['name'] = 'RandomForest'
    config['model']['n_estimators'] = 10
    config['model']['max_depth'] = 3
    config['paths']['model_path'] = "models/test_rf.pkl"
    metrics_rf = train_and_save_model(config)
    assert 'accuracy' in metrics_rf

    # Cobertura de Excepción (Modelo Inválido)
    config['model']['name'] = 'ModeloInexistente'
    with pytest.raises(ValueError):
        train_and_save_model(config)

    # Limpieza de binarios creados
    for p in ["test_lr.pkl", "test_rf.pkl"]:
        r = os.path.join(raiz_proyecto, "models", p)
        if os.path.exists(r):
            os.remove(r)
            
def test_train_and_save_model_file_not_found(config):
    # Forzar error de archivo no encontrado borrando temporalmente rutas conocidas
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            train_and_save_model(config)


# ==========================================
# PRUEBAS PARA API.PY Y PREDICT.PY
# ==========================================

def test_api_predict_endpoint_success(valid_payload):
    # Creamos un mock de un clasificador entrenado
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure']
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.8, 0.2]]

    with patch.object(api_module, 'model', mock_model):
        response = client_api.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["label"] == "No Churn"


def test_api_predict_no_model(valid_payload):
    with patch.object(api_module, 'model', None):
        response = client_api.post("/predict", json=valid_payload)
        assert response.status_code == 503


def test_api_predict_exception(valid_payload):
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("Fallo forzado simulado")
    
    with patch.object(api_module, 'model', mock_model):
        response = client_api.post("/predict", json=valid_payload)
        assert response.status_code == 400


def test_predict_app_endpoint_success(valid_payload):
    import src.predict as predict_module
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ['gender', 'Partner']
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.3, 0.7]]

    with patch.object(predict_module, 'model', mock_model):
        response = client_predict.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["label"] == "Churn"


# ==========================================
# PRUEBAS PARA MAIN.PY (ENTRYPOINT)
# ==========================================

def test_main_script_flows():
    # Flujo de error por falta de argumentos numéricos mínimos
    with patch.object(sys, 'argv', ['main.py']):
        with pytest.raises(SystemExit) as exc_info:
            main_entrypoint()
        assert exc_info.value.code == 1

    # Flujo de comando no reconocido
    with patch.object(sys, 'argv', ['main.py', 'comando_invalido']):
        with pytest.raises(SystemExit) as exc_info:
            main_entrypoint()
        assert exc_info.value.code == 1

    # Flujo Exitoso simulado del comando 'prepare'
    with patch.object(sys, 'argv', ['main.py', 'prepare']):
        with patch('src.main.load_and_preprocess_data', return_value=(pd.DataFrame(0, index=[0], columns=['a']), None, None, None)):
            main_entrypoint() # No debe crashear, solo ejecutar el print interno

    # Flujo Exitoso simulado del comando 'train'
    with patch.object(sys, 'argv', ['main.py', 'train']):
        with patch('src.main.train_and_save_model', return_value={'accuracy': 0.95}):
            main_entrypoint()