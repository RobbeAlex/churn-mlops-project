# 📊 Datos y Preprocesamiento

El combustible de este proyecto de Machine Learning es el dataset **Telco Customer Churn**. En este documento se detalla la naturaleza de estos datos y las transformaciones aplicadas durante la etapa de *Data Engineering*.

## Origen del Dataset

- **Nombre:** Telco Customer Churn
- **Fuente:** Dataset de muestra original de IBM, alojado en [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Problema de Negocio:** Una compañía de telecomunicaciones ficticia ha recolectado datos sobre los clientes y desea identificar quiénes son propensos a cancelar su contrato mensual (`Churn = Yes`).

## Descripción de las Variables

El dataset cuenta con 7,043 registros (clientes) y 21 variables, divididas en las siguientes categorías:

1. **Datos Demográficos:**
   - `gender` (Hombre/Mujer), `SeniorCitizen` (Tercera edad o no), `Partner` (Tiene pareja), `Dependents` (Tiene dependientes).
2. **Servicios Contratados:**
   - Servicios de teléfono, múltiples líneas, internet (DSL, Fibra, No), seguridad online, soporte técnico, streaming, etc.
3. **Información de Cuenta/Facturación:**
   - `tenure` (Meses de antigüedad con la compañía).
   - `Contract` (Mes a mes, Un año, Dos años).
   - `PaperlessBilling` (Facturación electrónica).
   - `PaymentMethod` (Transferencia, Tarjeta, Cheque electrónico, Cheque por correo).
   - `MonthlyCharges` (Monto mensual cobrado).
   - `TotalCharges` (Monto total cobrado al cliente históricamente).
4. **Target (Variable a Predecir):**
   - `Churn` (Yes / No). Indica si el cliente abandonó la empresa el mes pasado.

> **Ojo:** El dataset presenta un **fuerte desbalance de clases**, con aproximadamente un 73.5% de clientes retenidos (`No Churn`) y solo un 26.5% que abandonaron (`Churn`).

## Pasos de Preprocesamiento (`src/data_loader.py`)

Para que el modelo pueda interpretar correctamente estos datos, el pipeline ejecuta automáticamente una serie de transformaciones:

### 1. Limpieza de Datos
- **Manejo de strings vacíos:** La variable `TotalCharges` se lee como texto. En algunos casos, clientes con `tenure=0` tienen espacios en blanco en esta columna. Se reemplazan por el valor de `MonthlyCharges`.
- **Conversión de tipos:** `TotalCharges` se fuerza a tipo numérico (float).
- **Eliminación de identificadores:** La columna `customerID` se elimina, ya que no aporta poder predictivo y podría causar sobreajuste (overfitting).

### 2. Codificación de Variables Categóricas
La mayoría de algoritmos de Scikit-Learn requieren variables numéricas.
- Se utiliza `LabelEncoder` para transformar las variables de texto binarias o multicategóricas en variables numéricas (0, 1, 2, ...).
- La variable objetivo `Churn` se convierte a entero: `Yes -> 1`, `No -> 0`.

### 3. Escalamiento de Características Numéricas
- Se aplica un `StandardScaler` a todas las variables numéricas para estandarizar el rango de valores (media=0, varianza=1), lo cual mejora el rendimiento y estabilidad de muchos algoritmos de Machine Learning.

### 4. Partición del Dataset (Train/Test Split)
- Se separa el 20% del dataset para pruebas (testing) y el 80% para entrenamiento.
- Se utiliza `random_state=42` para garantizar la reproducibilidad de los splits.

> **Archivos Generados:** El script de preprocesamiento guarda los datos transformados en el directorio `data/processed/`, que es rastreado por DVC.
