# 🧠 Modelado y Experimentación

En el contexto de MLOps, el entrenamiento de un modelo no es un evento aislado, sino un proceso iterativo. Esta sección documenta el algoritmo seleccionado, sus métricas de rendimiento y cómo garantizamos la trazabilidad de los experimentos.

## El Modelo: Random Forest Classifier

Para resolver el problema de clasificación de `Churn` (cliente se va = 1, cliente se queda = 0), hemos seleccionado el algoritmo **Random Forest Classifier** de la librería Scikit-Learn (`sklearn.ensemble.RandomForestClassifier`).

### ¿Por qué Random Forest?
- **Robusto frente a ruido y valores atípicos:** Al ser un ensamble de múltiples árboles de decisión.
- **Interpretabilidad:** Permite obtener de forma directa la "Importancia de las Características" (*Feature Importance*), lo cual es crucial para el negocio (por ejemplo, saber que `Contract_Month-to-month` o `tenure` son determinantes en el abandono).
- **Manejo de desbalanceo (en parte):** Aunque el dataset está desbalanceado (74% vs 26%), los ensambles basados en árboles suelen lidiar mejor con esto que los modelos lineales clásicos.

## Métricas Obtenidas

El script `src/model_trainer.py` entrena el algoritmo y calcula el rendimiento sobre el set de pruebas (`X_test`, `y_test`).

| Métrica | Valor Obtenido | Interpretación de Negocio |
| :--- | :--- | :--- |
| **Accuracy** (Exactitud) | ~0.8155 | El modelo acierta el 81.55% de las veces en su predicción general. |
| **Recall** (Sensibilidad) | ~0.5389 | De todos los clientes que **realmente se van a dar de baja**, el modelo es capaz de identificar proactivamente a casi el 54%. |

### Análisis de los Resultados
- El **Accuracy** es alto y puede parecer bueno a simple vista, pero dado el desbalance de los datos (predecir "siempre que nadie se va" ya daría un 74% de accuracy), no es la mejor métrica.
- El **Recall** de 0.54 es la métrica más crítica en el problema del *Churn*. Significa que perdemos alrededor del 46% de los clientes en riesgo porque el modelo no los detecta. 
- *Trabajo futuro:* Podría aplicarse técnicas como **SMOTE** (Synthetic Minority Over-sampling Technique) para balancear las clases en el entrenamiento y mejorar el Recall.

## Tracking y Versionado con MLflow y DVC

Para evitar la pérdida de contexto entre entrenamientos ("*¿Qué hiperparámetros me dieron mejor Recall hace una semana?*"), utilizamos dos herramientas de MLOps de estándar en la industria:

### DVC (Data Version Control)
Orquesta el flujo. Al correr `dvc repro`, DVC verifica si hubo cambios en los datos o en los scripts definidos en `dvc.yaml`. Si los hay, ejecuta la secuencia correcta.

### MLflow
Automatiza el registro de experimentos:
- En `src/model_trainer.py`, abrimos un contexto de MLflow (`mlflow.start_run()`).
- Se registran automáticamente los parámetros del modelo (ej. `n_estimators`, `max_depth`) usando `mlflow.sklearn.autolog()`.
- Se registran las métricas personalizadas (Accuracy, Recall) usando `mlflow.log_metric()`.
- Se almacena el archivo binario del modelo en la carpeta `models/` usando `joblib`.

### Visualización en MLflow UI
Puedes ver el histórico de los entrenamientos iniciando el servidor local de MLflow:
```bash
mlflow ui
```
En `http://localhost:5000` podrás comparar visualmente cada ejecución y decidir qué versión del modelo promover a producción.
