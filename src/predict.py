import joblib
import pandas as pd
import yaml
import sys
import os

# 1. Asegurar que Python reconozca la carpeta raíz sin importar cómo se ejecute el script
ruta_script = os.path.abspath(__file__)
directorio_src = os.path.dirname(ruta_script)
raiz_proyecto = os.path.dirname(directorio_src)

if raiz_proyecto not in sys.path:
    sys.path.append(raiz_proyecto)

# Ahora la importación funcionará siempre
from src.data_loader import load_and_preprocess_data


def load_config():
    # 2. Construir la ruta absoluta hacia el archivo YAML
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')
    try:
        with open(ruta_config, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error crítico: No se encontró el archivo de configuración en '{ruta_config}'.")
        sys.exit(1)


def predict_new_customer(customer_data_df):
    config = load_config()

    # 3. Blindar también la ruta donde se guarda el modelo
    ruta_relativa_modelo = config['paths']['model_save'].replace('/', os.sep)
    model_path = os.path.join(raiz_proyecto, ruta_relativa_modelo)

    # Manejo de errores si el modelo no existe
    if not os.path.exists(model_path):
        print(f"Error crítico: No se encontró un modelo entrenado en '{model_path}'.")
        print("Por favor, ejecuta 'python -m src.main' primero para entrenarlo.")
        sys.exit(1)

    # Cargar modelo e inferir
    model = joblib.load(model_path)
    prediction = model.predict(customer_data_df)

    return "Churn (Sí abandona)" if prediction[0] == 1 else "No Churn (Se queda)"


if __name__ == "__main__":
    config = load_config()

    print("Cargando esquema de datos para emparejar formato del cliente de prueba...")
    X_train, _, _, _ = load_and_preprocess_data(config)

    # Crear un cliente de prueba con ceros en todas las columnas categóricas
    sample_customer = pd.DataFrame(columns=X_train.columns)
    sample_customer.loc[0] = 0

    # Asignar valores específicos numéricos y categóricos preprocesados
    sample_customer['tenure'] = 2
    sample_customer['MonthlyCharges'] = 70.5
    sample_customer['TotalCharges'] = 141.0
    sample_customer['gender'] = 1  # Female
    sample_customer['Partner'] = 0  # No

    print("\nEjecutando predicción para el cliente de ejemplo...")
    result = predict_new_customer(sample_customer)
    print(f"Resultado de la predicción: {result}")