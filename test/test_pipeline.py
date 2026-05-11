import pytest
import yaml
import os
import sys

# Agregar la raíz al sys.path para que PyCharm y pytest encuentren 'src'
ruta_actual = os.path.abspath(os.path.dirname(__file__))
raiz_proyecto = os.path.dirname(ruta_actual)
if raiz_proyecto not in sys.path:
    sys.path.append(raiz_proyecto)

from src.data_loader import load_and_preprocess_data
# Importación corregida para coincidir con tu archivo físico
from src.trainer_model import train_and_save_model


@pytest.fixture
def config():
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')
    with open(ruta_config, 'r') as file:
        return yaml.safe_load(file)


def test_load_and_preprocess_data_not_empty(config):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    assert X_train is not None and not X_train.empty
    assert X_test is not None and not X_test.empty
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_train_and_save_model_returns_metrics(config):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    config['model']['name'] = 'LogisticRegression'
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics