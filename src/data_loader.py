import pandas as pd
from sklearn.model_selection import train_test_split
import os


def load_and_preprocess_data(config):
    """Carga y preprocesa los datos según los parámetros de configuración."""

    # 1. Construir la ruta absoluta de forma dinámica
    ruta_script = os.path.abspath(__file__)
    directorio_src = os.path.dirname(ruta_script)
    raiz_proyecto = os.path.dirname(directorio_src)

    # 2. Extraer la ruta relativa del YAML y unirla con la raíz del proyecto
    # Usamos replace para evitar problemas entre diagonales de Windows (\) y Linux/Mac (/)
    ruta_relativa_csv = config['paths']['data_raw'].replace('/', os.sep)
    ruta_csv_absoluta = os.path.join(raiz_proyecto, ruta_relativa_csv)

    # 3. Cargar el CSV usando la ruta blindada
    df = pd.read_csv(ruta_csv_absoluta)

    # Limpiar TotalCharges: convertir a numérico, reemplazar espacios por NaN e imputar con la mediana
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', pd.NA))
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Eliminar la columna customerID
    df = df.drop(columns=['customerID'])

    # Codificar variables categóricas principales a 0/1
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Transformar el resto de variables categóricas en dummies para que el modelo no falle
    X = df.drop(columns=['Churn'])
    X = pd.get_dummies(X, drop_first=True)
    y = df['Churn']

    # Dividir el dataset usando los parámetros de configuración
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    return X_train, X_test, y_train, y_test