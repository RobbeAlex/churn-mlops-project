import joblib
import pandas as pd
import yaml
import sys
import os

# 1. Ruta robusta a la raíz del proyecto
ruta_script = os.path.abspath(__file__)
directorio_src = os.path.dirname(ruta_script)
raiz_proyecto = os.path.dirname(directorio_src)

if raiz_proyecto not in sys.path:
    sys.path.append(raiz_proyecto)

# 2. Importar función de procesamiento de datos
from src.data_loader import load_and_preprocess_data

def load_config():
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')
    try:
        with open(ruta_config, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error crítico: No se encontró el archivo de configuración en '{ruta_config}'.")
        sys.exit(1)

def predict_new_customer(customer_data_dict):
    """
    Recibe un diccionario con las llaves crudas (tipo API), procesa las features,
    alinea las columnas y devuelve la predicción.
    """
    config = load_config()

    # Obtener estructura de columnas de entrenamiento real
    X_train, _, _, _ = load_and_preprocess_data(config)
    all_columns = X_train.columns

    # Convertir el dict de cliente a DataFrame de una fila
    cliente_df = pd.DataFrame([customer_data_dict])

    # Unir para asegurar el correcto procesado como en entrenamiento
    df_unido = pd.concat([X_train, cliente_df], ignore_index=True)
    df_unido = pd.get_dummies(df_unido)

    # Tomar SOLO la última fila (cliente nuevo)
    cliente_procesado = df_unido.tail(1)

    # Asegurarse de que las columnas estén en el mismo orden y presencia que el entrenamiento
    cliente_procesado = cliente_procesado.reindex(columns=all_columns, fill_value=0)

    # Cargar modelo
    config_path = config['paths']['model_path']
    if config_path.startswith('/app/'):
        config_path = config_path.replace('/app/', '', 1)
    elif config_path.startswith('/'):
        config_path = config_path.lstrip('/')
    ruta_modelo = os.path.join(raiz_proyecto, config_path.replace('/', os.sep))
    if not os.path.exists(ruta_modelo):
        print(f"Error crítico: No se encontró un modelo entrenado en '{ruta_modelo}'.")
        print("Por favor, ejecuta 'python -m src.main' primero para entrenarlo.")
        sys.exit(1)

    model = joblib.load(ruta_modelo)

    pred = model.predict(cliente_procesado)[0]
    prob = model.predict_proba(cliente_procesado)[0][1]

    return {
        "prediction": int(pred),
        "label": "Churn (Sí abandona)" if pred == 1 else "No Churn (Se queda)",
        "probability": round(float(prob), 2)
    }

if __name__ == "__main__":
    # Ejemplo de uso
    ejemplo_cliente = {
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

    resultado = predict_new_customer(ejemplo_cliente)
    print("Resultado de la predicción:", resultado)
