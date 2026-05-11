# 📡 Proyecto Colaborativo MLOps: Predicción de Churn (Abandono de Clientes)

## 🎯 Objetivo del Proyecto
Construir un pipeline de Machine Learning modular, reproducible y colaborativo para predecir si un cliente de telecomunicaciones abandonará el servicio (**Churn**).

El proyecto simula un entorno laboral real donde **4 roles especializados** deben integrar su código en un solo repositorio usando Git.

---

## 📂 El Dataset
Todos los equipos trabajarán con el dataset **Telco Customer Churn**.

*   **Fuente:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
*   **Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
*   **Problema:** Clasificación Binaria (¿El cliente se va? `Yes`/`No`)
*   **Instrucción Importante:**
    1.  Descarguen el CSV.
    2.  Guárdenlo en la carpeta `data/raw/`.
    3.  **NO suban el CSV a Git** (ya está configurado en `.gitignore` para evitar subir archivos pesados). Cada alumno debe descargarlo localmente.

---

## 👥 Roles y Responsabilidades (Equipos de 4)

Cada miembro del equipo es responsable de un módulo específico. Deben definir sus "contratos de interface" (nombres de funciones y tipos de datos que pasan entre módulos) antes de empezar a codificar.

### 1. 👷 Data Engineer (`src/data_loader.py`)
**Tu misión:** Transformar datos brutos y sucios en datos limpios listos para entrenar.

*   **Tareas Críticas:**
    *   Cargar el CSV desde `data/raw/`.
    *   **Limpieza:** La columna `TotalCharges` tiene espacios vacíos `" "` en lugar de nulos. Debes convertirla a numérico y manejar los NaN resultantes (ej. llenar con mediana o 0).
    *   **Preprocesamiento:** Eliminar `customerID`. Codificar variables binarias (`gender`, `Partner`, `Churn`) de Texto a 0/1.
    *   **División:** Separar en Train/Test usando `test_size` y `random_state` definidos en `config/params.yaml`.
*   **Entregable:** Función `load_and_preprocess_data(config)` que retorna `X_train, X_test, y_train, y_test`.

### 2. 🧠 ML Engineer (`src/model_trainer.py`)
**Tu misión:** Experimentar con algoritmos y guardar el mejor modelo.

*   **Tareas Críticas:**
    *   Implementar una "Fábrica de Modelos" que permita elegir entre al menos **dos algoritmos** (ej. `RandomForest` y `SVM` o `LogisticRegression`) según el config.
    *   Entrenar el modelo con los datos recibidos.
    *   Calcular métricas clave: **Accuracy**, **Recall** (crítico para Churn) y **F1-Score**.
    *   Guardar el modelo entrenado en la carpeta `models/` usando `joblib`.
*   **Entregable:** Función `train_and_save_model(X_train, y_train, X_test, y_test, config)` que guarda el `.pkl` y retorna un diccionario de métricas.

### 3. ⚙️ MLOps Engineer (`src/main.py` y `config/`)
**Tu misión:** Orquestar el flujo y gestionar la configuración externa.

*   **Tareas Críticas:**
    *   Crear y mantener `config/params.yaml`. Debe incluir:
        *   Parámetros de datos (`test_size`, `random_state`).
        *   Parámetros del modelo (`model_name`, `n_estimators`, `C`, `kernel`, etc.).
        *   Rutas de salida.
    *   Escribir `src/main.py`: Este script debe importar las funciones del Data Engineer y del ML Engineer y ejecutarlas en orden.
    *   Asegurar que el proyecto corra con el comando: `python -m src.main`.
*   **Entregable:** Un `main.py` funcional que lea el YAML y ejecute el pipeline completo sin errores de importación.

### 4. 🛡️ QA & Production Engineer (`src/predict.py` y `tests/`)
**Tu misión:** Validar que el sistema funcione y preparar la inferencia para nuevos datos.

*   **Tareas Críticas:**
    *   Crear `src/predict.py`: Un script que cargue el modelo guardado (`models/model.pkl`) y permita predecir la clase de un nuevo cliente (ej. pasando una lista de características manualmente).
    *   Manejo de Errores: Si el modelo no existe, el script debe dar un mensaje claro, no un error críptico.
    *   Escribir tests básicos en `tests/test_pipeline.py` (ej. verificar que `load_data` no retorne DataFrames vacíos).
*   **Entregable:** Un script de predicción robusto y al menos 2 tests unitarios pasando.

---

## 🚀 Flujo de Trabajo con Git

1.  **Clonar:** `git clone <url-del-repo-del-equipo>`
2.  **Ramas:** Cada alumno crea su rama:
    *   `git checkout -b feature/data-engineer`
    *   `git checkout -b feature/ml-engineer`
    *   `git checkout -b feature/mlops-engineer`
    *   `git checkout -b feature/qa-engineer`
3.  **Desarrollo:** Trabajen en paralelo. Hagan commits frecuentes.
4.  **Integración:**
    *   Cuando terminen, hagan `git push` de sus ramas.
    *   El **MLOps Engineer** debe crear un Pull Request (o merge) integrando todas las ramas a `main`.
    *   **Resuelvan conflictos juntos** si dos personas tocaron el mismo archivo (ej. `requirements.txt` o `main.py`).
5.  **Prueba Final:** Ejecuten `python -m src.main` en la rama `main`. Si corre, ¡misión cumplida!

---

## 📂 Estructura de Carpetas

```text
churn-mlops-project/
├── config/
│   └── params.yaml          # Configuración centralizada
├── data/
│   ├── raw/                 # WA_Fn-UseC_-Telco-Customer-Churn.csv (NO SUBIR)
│   └── processed/           # (Opcional) Datos limpios
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Rol: Data Engineer
│   ├── model_trainer.py     # Rol: ML Engineer
│   ├── main.py              # Rol: MLOps Engineer
│   └── predict.py           # Rol: QA Engineer
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Rol: QA Engineer
├── models/                  # Modelos .pkl generados (NO SUBIR o subir solo el final)
├── requirements.txt         # Dependencias
├── .gitignore               # Reglas de exclusión
└── README.md                # Este archivo
```

---

## ✅ Checklist de Entrega

*   [ ] El comando `python -m src.main` ejecuta todo el pipeline sin errores.
*   [ ] El archivo `config/params.yaml` existe y controla los hiperparámetros.
*   [ ] Hay al menos 2 modelos diferentes implementados en el código.
*   [ ] El script `predict.py` carga el modelo y hace una predicción de ejemplo.
*   [ ] El historial de Git muestra contribuciones de los 4 miembros del equipo.
*   [ ] El `README.md` final incluye los resultados obtenidos (Accuracy/Recall del mejor modelo).



¡Éxito con la clase! Es un ejercicio excelente para ver quién realmente entiende la integración de sistemas. 🚀
