import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ==========================================
# 1. FUNCIÓN DE CARGA Y PREPROCESAMIENTO
# ==========================================
def load_and_preprocess_data(config):
    """Carga y preprocesa los datos según los parámetros de configuración."""

    # Construir la ruta absoluta de forma dinámica
    ruta_script = os.path.abspath(__file__)
    directorio_src = os.path.dirname(ruta_script)
    raiz_proyecto = os.path.dirname(directorio_src)

    # Extraer la ruta relativa del YAML y unirla con la raíz del proyecto
    ruta_relativa_csv = config['paths']['raw_data'].replace('/', os.sep)
    ruta_csv_absoluta = os.path.join(raiz_proyecto, ruta_relativa_csv)

    # Cargar el CSV usando la ruta blindada
    df = pd.read_csv(ruta_csv_absoluta)

    # Limpiar TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', pd.NA))
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Eliminar la columna customerID
    df = df.drop(columns=['customerID'])

    # Codificar variables categóricas principales a 0/1
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Transformar el resto de variables categóricas en dummies
    X = df.drop(columns=['Churn'])
    X = pd.get_dummies(X, drop_first=True)
    y = df['Churn']

    # === PARÁMETROS DE CONFIGURACIÓN SEGUROS ===
    test_size_config = config['data_split']['test_size']
    random_state = config['data_split']['random_state']

    # Dividir el dataset usando ambos parámetros y estratificando el target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size_config,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ==========================================
# 2. FUNCIÓN DE ENTRENAMIENTO Y GUARDADO
# ==========================================
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
    print("Columnas esperadas por el modelo:")
    print(list(model.feature_names_in_))  # ← aquí
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
