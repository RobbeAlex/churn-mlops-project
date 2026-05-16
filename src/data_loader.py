import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(config):
    """Carga, preprocesa los datos y los guarda en archivos procesados para el pipeline."""

    # 1. Construir la ruta absoluta de forma dinámica
    ruta_script = os.path.abspath(__file__)
    directorio_src = os.path.dirname(ruta_script)
    raiz_proyecto = os.path.dirname(directorio_src)

    # 2. Obtener rutas desde la configuración yaml
    ruta_relativa_csv = config['paths']['raw_data'].replace('/', os.sep)
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

    # Transformar el resto de variables categóricas en dummies
    X = df.drop(columns=['Churn'])
    X = pd.get_dummies(X, drop_first=True)
    y = df['Churn']

    # Dividir el dataset asegurando la estratificación del target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data_split']['test_size'],
        random_state=config['data_split']['random_state'],
        stratify=y
    )

    # Combinar características y targets para guardarlos de forma estructurada
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Definir rutas de guardado dentro de data/processed/
    processed_dir = os.path.join(raiz_proyecto, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"   [DVC DATA] Conjuntos de datos guardados exitosamente en: {processed_dir}")
    return X_train, X_test, y_train, y_test
