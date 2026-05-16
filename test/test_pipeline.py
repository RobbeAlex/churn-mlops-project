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
    Prueba la lógica de entrenamiento usando la nueva firma de función compatible con DVC.
    Alinea las rutas de guardado para simular con precisión el entorno de ejecución.
    """
    # 1. Ejecutar el procesado inicial para extraer la estructura de los datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    # 2. Reducir los datos para que el test unitario en el CI/CD sea veloz
    X_train_sub, y_train_sub = X_train.head(50), y_train.head(50)
    X_test_sub, y_test_sub = X_test.head(10), y_test.head(10)

    # Combinar características y target simulando el comportamiento de data_loader
    train_df = pd.concat([X_train_sub, y_train_sub], axis=1)
    test_df = pd.concat([X_test_sub, y_test_sub], axis=1)

    # 3. Escribir un lote de control rápido directo en el área de procesamiento local 
    # para que model_trainer lo consuma de forma nativa sin romper el flujo del disco
    real_processed_dir = os.path.join(raiz_proyecto, 'data', 'processed')
    os.makedirs(real_processed_dir, exist_ok=True)
    
    test_train_path = os.path.join(real_processed_dir, 'train.csv')
    test_test_path = os.path.join(real_processed_dir, 'test.csv')
    
    train_df.to_csv(test_train_path, index=False)
    test_df.to_csv(test_test_path, index=False)

    # 4. Modificar la configuración dinámica del test
    config['model']['name'] = 'LogisticRegression'
    
    # IMPORTANTE: Usamos una ruta relativa para que no choque con el os.path.join(raiz_proyecto, ...) de model_trainer
    nombre_modelo_test = "test_model_pipeline.pkl"
    config['paths']['model_path'] = f"models/{nombre_modelo_test}"

    # 5. Ejecutar entrenamiento con la firma correcta de un solo argumento
    metrics = train_and_save_model(config)

    # 6. Calcular con precisión milimétrica la ruta donde model_trainer guardó el archivo
    ruta_esperada_modelo = os.path.join(raiz_proyecto, 'models', nombre_modelo_test)

    # Validaciones del entrenamiento
    assert metrics is not None, "La función de entrenamiento debería retornar el diccionario de métricas."
    assert 'accuracy' in metrics, "La clave 'accuracy' no está presente en las métricas resultantes"
    assert os.path.exists(ruta_esperada_modelo), f"El modelo binario no se encontró en la ruta calculada de producción: {ruta_esperada_modelo}"
    
    # Limpieza post-test: Evita dejar archivos temporales en tu carpeta 'models/' local o de GitHub Actions
    if os.path.exists(ruta_esperada_modelo):
        os.remove(ruta_esperada_modelo)


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
