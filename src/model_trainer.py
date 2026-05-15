import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train_and_save_model(X_train, y_train, X_test, y_test, config):
    model_name = config['model']['name']
    random_state = config['data_split']['random_state']

    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=config['model'].get('n_estimators', 100),
            max_depth=config['model'].get('max_depth', None),
            random_state=random_state
        )
    elif model_name == 'LogisticRegression':
        # Buscamos 'max_iter' en la configuración; si no se define, usamos 2000 por defecto
        max_iter_config = config['model'].get('max_iter', 2000)
        model = LogisticRegression(random_state=random_state, max_iter=max_iter_config)
    else:
        raise ValueError(f"El modelo '{model_name}' no está soportado.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    # === SISTEMA DE RUTAS SEGURO Y BLINDADO ===
    # 1. Extraemos correctamente la ruta desde la sección 'paths' del diccionario
    config_path = config['paths']['model_path']

    if os.path.isabs(config_path):
        save_path = config_path
    else:
        if config_path.startswith('/app/'):
            config_path = config_path.replace('/app/', '', 1)
        elif config_path.startswith('/'):
            config_path = config_path.lstrip('/')

        ruta_script = os.path.abspath(__file__)
        raiz_proyecto = os.path.dirname(os.path.dirname(ruta_script))
        save_path = os.path.join(raiz_proyecto, config_path.replace('/', os.sep))

    # IMPRIMIR LA RUTA REAL EN LA CONSOLA DE PYTEST
    print(f"\n[DEBUG TRAIN] El modelo se intentará guardar en: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

    return metrics