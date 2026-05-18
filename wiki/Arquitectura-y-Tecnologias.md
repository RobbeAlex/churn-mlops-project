# 🏗️ Arquitectura y Tecnologías

Para garantizar un ciclo de vida de Machine Learning (ML) robusto, colaborativo y reproducible, este proyecto ha implementado una arquitectura MLOps moderna que integra el código fuente, la manipulación de datos, el entrenamiento del modelo y su despliegue.

## Diagrama de la Arquitectura

<p align="center">
  <img src="https://raw.githubusercontent.com/RobbeAlex/churn-mlops-project/main/docs/images/Arquitectura%20de%20Pipeline.png" alt="Arquitectura del Pipeline MLOps" width="100%">
</p>

## Flujo de Trabajo (Pipeline)

El pipeline de MLOps de este proyecto se divide en las siguientes etapas clave:

1. **Ingesta y Preprocesamiento (Data Engineering)**
   - **Script:** `src/data_loader.py`
   - Los datos crudos (almacenados y versionados en DVC) son cargados y limpiados (imputación de nulos, eliminación de columnas irrelevantes como `customerID`).
   - Se dividen en conjuntos de *Train* y *Test*.
   - Se aplican transformaciones categóricas y escalamiento.

2. **Entrenamiento y Evaluación (ML Engineering)**
   - **Script:** `src/model_trainer.py`
   - Se entrena el algoritmo (por defecto, Random Forest Classifier) utilizando los datos procesados.
   - Durante este proceso, se registran las métricas clave (*Accuracy*, *Recall*) y los artefactos del modelo directamente en **MLflow**.

3. **Orquestación y Versionado (MLOps)**
   - **Herramienta:** `dvc` (Data Version Control)
   - El archivo `dvc.yaml` define cómo se relacionan los scripts y los datos, creando un DAG (Grafo Acíclico Dirigido). Si los datos cambian, `dvc repro` sabe exactamente qué partes del pipeline necesitan volver a ejecutarse.

4. **Despliegue e Inferencia (Producción)**
   - **Script:** `src/api.py`
   - Una vez entrenado, el modelo (`model.joblib`) es cargado en una aplicación web ligera usando **FastAPI**.
   - Se expone un endpoint (`/predict`) que acepta un JSON con características de clientes y devuelve la predicción.
   - Toda la aplicación se empaqueta en una imagen de **Docker** para asegurar que el entorno de ejecución sea idéntico en desarrollo, pruebas y producción.

---

## Stack Tecnológico 🛠️

| Componente | Herramienta/Librería | Propósito en el Proyecto |
| :--- | :--- | :--- |
| **Lenguaje Base** | Python 3.9+ | Lenguaje principal del proyecto. |
| **Desarrollo Web / API** | FastAPI, Uvicorn | Creación del servidor web y endpoints RESTful asíncronos y rápidos. |
| **Machine Learning** | scikit-learn, pandas | Manipulación de datos y creación del algoritmo de clasificación predictivo. |
| **Control de Versiones de Datos**| DVC | Versionar el dataset gigante y orquestar el pipeline de procesamiento/entrenamiento. |
| **Tracking de Experimentos** | MLflow | Almacenamiento de hiperparámetros y métricas de diferentes ejecuciones. |
| **Contenedores** | Docker, Docker Compose | Empaquetado de la API y el entorno en contenedores reproducibles. |
| **Testing y Calidad** | pytest, pre-commit, Codecov | Ejecución de pruebas automatizadas y estandarización del código (linting). |
| **Gestión de Entornos** | venv, requirements.txt | Aislamiento de dependencias de Python de forma local. |
