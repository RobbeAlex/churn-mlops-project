import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score


def train_and_save_model(config):
    """Carga los datos procesados por DVC, entrena el modelo seleccionado y guarda los artefactos."""
    
    ruta_script = os.path.abspath(__file__)
    raiz_proyecto = os.path.dirname(os.path.dirname(ruta_script))
    
    # Cargar los CSV procesados en la etapa anterior
    train_path = os.path.join(raiz_proyecto, 'data', 'processed', 'train.csv')
    test_path = os.path.join(raiz_proyecto, 'data', 'processed', 'test.csv')
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("No se encontraron los datos procesados en data/processed/. Ejecuta primero la etapa de preparación.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    model_name = config['model']['name']
    random_state = config['data_split']['random_state']

    mlflow.set_experiment("Prediccion_Churn_Telco")

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("random_state", random_state)

        if model_name == 'RandomForest':
            n_estimators = config['model'].get('n_estimators', 100)
            max_depth = config['model'].get('max_depth', None)
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        elif model_name == 'LogisticRegression':
            max_iter_config = config['model'].get('max_iter', 2000)
            mlflow.log_param("max_iter", max_iter_config)
            model = LogisticRegression(random_state=random_state, max_iter=max_iter_config)
        else:
            raise ValueError(f"El modelo '{model_name}' no está soportado.")

        # Entrenamiento (Aquí se almacena internamente model.feature_names_in_)
        model.fit(X_train, y_train)
        
        # Predicción y Métricas
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "modelo_churn")

        # Configuración robusta de guardado para el archivo .pkl
        config_path = config['paths']['model_path']
        if config_path.startswith('/app/'):
            config_path = config_path.replace('/app/', '', 1)
        elif config_path.startswith('/'):
            config_path = config_path.lstrip('/')
            
        save_path = os.path.join(raiz_proyecto, config_path.replace('/', os.sep))
        print(f"\n[DEBUG TRAIN] Guardando modelo en: {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

    return metrics
