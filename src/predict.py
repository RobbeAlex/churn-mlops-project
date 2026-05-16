import os
import sys
import joblib
import pandas as pd
import yaml

ruta_script = os.path.abspath(__file__)
directorio_src = os.path.dirname(ruta_script)
raiz_proyecto = os.path.dirname(directorio_src)

def load_config():
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')
    try:
        with open(ruta_config, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error crítico: No se encontró el archivo de configuración en '{ruta_config}'.")
        sys.exit(1)

def predict_new_customer(customer_data_dict):
    config = load_config()

    # Cargar el modelo entrenado
    config_path = config['paths']['model_path']
    if config_path.startswith('/app/'):
        config_path = config_path.replace('/app/', '', 1)
    elif config_path.startswith('/'):
        config_path = config_path.lstrip('/')
    ruta_modelo = os.path.join(raiz_proyecto, config_path.replace('/', os.sep))
    
    if not os.path.exists(ruta_modelo):
        print(f"Error crítico: No se encontró un modelo entrenado en '{ruta_modelo}'.")
        sys.exit(1)

    model = joblib.load(ruta_modelo)

    # Convertir datos crudos del usuario a DataFrame
    cliente_df = pd.DataFrame([customer_data_dict])

    # Aplicar exactamente el mismo mapeo binario manual que en el entrenamiento
    cliente_df['gender'] = cliente_df['gender'].map({'Female': 1, 'Male': 0})
    cliente_df['Partner'] = cliente_df['Partner'].map({'Yes': 1, 'No': 0})
    if 'Churn' in cliente_df.columns:
        cliente_df = cliente_df.drop(columns=['Churn'])

    # Aplicar One-Hot Encoding para las variables de texto restantes
    cliente_procesado = pd.get_dummies(cliente_df)

    # ALINEACIÓN ULTRA-EFICIENTE usando la metadata nativa del modelo entrenado
    if hasattr(model, "feature_names_in_"):
        all_columns = model.feature_names_in_
        cliente_procesado = cliente_procesado.reindex(columns=all_columns, fill_value=0)
    else:
        print("Advertencia: El modelo no contiene el atributo 'feature_names_in_'.")

    pred = model.predict(cliente_procesado)[0]
    prob = model.predict_proba(cliente_procesado)[0][1]

    return {
        "prediction": int(pred),
        "label": "Churn" if pred == 1 else "No Churn",
        "probability": round(float(prob), 2)
    }

if __name__ == "__main__":
    ejemplo_cliente = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
        "tenure": 5, "PhoneService": 1, "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": 1, "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0, "TotalCharges": 400.0
    }
    print("Resultado de la predicción de prueba:", predict_new_customer(ejemplo_cliente))
