import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train_and_save_model(X_train, y_train, X_test, y_test, config):
    model_name = config['model']['name']
    random_state = config['data']['random_state']

    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=config['model'].get('n_estimators', 100),
            max_depth=config['model'].get('max_depth', None),
            random_state=random_state
        )
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    else:
        raise ValueError(f"El modelo '{model_name}' no está soportado.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    # Guardado blindado en la raíz del proyecto
    ruta_script = os.path.abspath(__file__)
    raiz_proyecto = os.path.dirname(os.path.dirname(ruta_script))
    ruta_relativa_modelo = config['paths']['model_save'].replace('/', os.sep)
    save_path = os.path.join(raiz_proyecto, ruta_relativa_modelo)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

    return metrics