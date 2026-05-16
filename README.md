# 📡 Proyecto MLOps: Predicción de Churn en Telecomunicaciones

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-purple)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Python application](https://github.com/RobbeAlex/churn-mlops-project/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/RobbeAlex/churn-mlops-project/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/RobbeAlex/churn-mlops-project/graph/badge.svg?token=WFGYQV4VB3)](https://codecov.io/gh/RobbeAlex/churn-mlops-project)

Pipeline de Machine Learning modular y reproducible para predecir el abandono de clientes (**Churn**) en telecomunicaciones. El proyecto simula un entorno colaborativo con **4 roles de ingeniería** aplicando buenas prácticas de MLOps: desde la ingesta de datos hasta el despliegue de una API REST. Incorpora **DVC** para el control de versiones de datos, **MLflow** para tracking de experimentos, y herramientas de CI/CD (**GitHub Actions, Codecov, pre-commit**) para asegurar la máxima calidad de código.

---

## 📋 Tabla de Contenidos

- [¿Qué hace este proyecto?](#-qué-hace-este-proyecto)
- [Resultados del Modelo](#-resultados-del-modelo)
- [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Cómo Ejecutar](#-cómo-ejecutar)
- [Uso de la API](#-uso-de-la-api)
- [Roles y Responsabilidades](#-roles-y-responsabilidades)
- [Flujo de Trabajo y Calidad de Código](#-flujo-de-trabajo-y-calidad-de-código)

---

## 🎯 ¿Qué hace este proyecto?

Desarrolla un **sistema completo de predicción de abandono de clientes** que incluye:

- **Preprocesamiento automático** (imputación, encoding, split train/test).
- **Entrenamiento de modelos** (Random Forest / Regresión Logística).
- **Tracking de Experimentos** usando MLflow (registro de métricas y parámetros).
- **Control de Versiones de Datos (DVC)** para rastrear datasets y pipelines.
- **Calidad de Código y Cobertura** continua (pre-commit, Codecov, GitHub Actions).
- **API REST con FastAPI** lista para producción.
- **Contenerización con Docker** para máxima reproducibilidad.

---

## 🏆 Resultados del Modelo

**Algoritmo seleccionado:** `RandomForestClassifier` (100 estimadores, max_depth=10)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 0.8155 | 81.55% de predicciones correctas globales. |
| **Recall** | 0.5389 | Detecta el 53.89% de los clientes que realmente abandonan. |
| **F1-Score** | 0.6073 | Balance adecuado entre precisión y exhaustividad. |

> **⚠️ Nota:** El desbalance de clases (74% No Churn / 26% Churn) afecta el Recall. Para aplicaciones críticas de negocio, se sugiere aplicar técnicas como SMOTE o ajustar `class_weight`.

---

## 🛠️ Tecnologías Utilizadas

- **Core & ML:** Python (3.9+), scikit-learn, pandas, numpy.
- **MLOps:** MLflow (Experiment Tracking), DVC (Data Versioning).
- **API & Despliegue:** FastAPI, Uvicorn, Docker, Docker Compose.
- **Calidad y Testing:** pytest, pre-commit (Black, Ruff), Codecov, GitHub Actions.

---

## 🚀 Instalación

Asegúrate de contar con Python 3.9+, Git, y Docker (opcional). Puedes descargar el dataset manualmente en `data/raw/` (ver `DATASET.md` para más información).

### Opción A: Entorno Local

```bash
# 1. Clonar el repositorio y crear entorno virtual
git clone https://github.com/RobbeAlex/churn-mlops-project.git
cd churn-mlops-project
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Instalar dependencias y hooks de calidad de código
pip install -r requirements.txt
pre-commit install

# 3. Descargar el dataset (si no usas DVC)
python scripts/download_dataset.py
```

### Opción B: Docker (Recomendado)

```bash
git clone https://github.com/RobbeAlex/churn-mlops-project.git
cd churn-mlops-project
docker-compose build
```

---

## 📂 Estructura del Proyecto

```text
churn-mlops-project/
├── .github/workflows/          # Pipelines CI/CD en GitHub Actions
├── config/                     # Configuraciones (params.yaml)
├── data/                       # Datos raw y procesados (gestionados por DVC)
├── docs/                       # Documentación adicional (DATASET.md, ETHICS.md)
├── models/                     # Modelos serializados (.pkl)
├── scripts/                    # Scripts auxiliares (descarga de dataset, etc)
├── src/                        # Código fuente del pipeline y la API (FastAPI)
├── test/                       # Tests unitarios con pytest
├── dvc.yaml                    # Pipeline de Data Version Control
├── codecov.yml                 # Configuración de cobertura
├── .pre-commit-config.yaml     # Hooks de linting y formateo (Black, Ruff)
├── docker-compose.yml          # Orquestación de contenedores
└── requirements.txt            # Dependencias del proyecto
```

---

## 🏃 Cómo Ejecutar

### 1. Ejecutar el Pipeline ML (DVC)
Utilizamos DVC para orquestar la preparación de datos y el entrenamiento:
```bash
dvc repro
```

### 2. Ver Métricas y Experimentos (MLflow)
Inicia la interfaz de MLflow para revisar el historial de modelos:
```bash
mlflow ui  # Disponible en http://127.0.0.1:5000
```

### 3. Levantar la API
- **Local (Uvicorn):** `uvicorn src.api:app --reload`
- **Docker:** `docker-compose up -d api`

---

## 🌐 Uso de la API

La API cuenta con endpoints para revisión de estado (`GET /health`) y para predicciones (`POST /predict`).

**Ejemplo de Petición (POST):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"tenure": 2, "MonthlyCharges": 70.5, "TotalCharges": 141.0, "gender": 1, "InternetService": 2, "Contract": 0, "PaymentMethod": 2}'
```

> **Documentación Interactiva:** Visita `http://127.0.0.1:8000/docs` con la API corriendo.

---

## 👥 Roles y Responsabilidades

- **Data Engineer:** Limpieza, encoding, y gestión de dependencias de datos en DVC (`src/data_loader.py`).
- **ML Engineer:** Entrenamiento, métricas, y logging en MLflow (`src/model_trainer.py`).
- **MLOps Engineer:** Orquestación (`dvc.yaml`, `config/`), Docker, e infraestructura.
- **QA & Prod Engineer:** Pruebas unitarias (`test/`), pre-commit hooks, CI/CD con Codecov y despliegue de la API (`src/api.py`).

---

## ✅ Flujo de Trabajo y Calidad de Código

Este proyecto utiliza el flujo de Git por ramas de funcionalidad (`feature/tu-rol`) integrado fuertemente con herramientas de calidad.

- **Calidad Continua (pre-commit):** Formateo automático y validación antes de cada commit mediante Black y Ruff.  
  `pre-commit run --all-files`
- **Testing y Cobertura (Codecov):** Tests unitarios garantizados en CI.  
  `python -m pytest test/ --cov=src --cov-report=html`

---

> **Consideraciones Éticas:** Para un desglose exhaustivo sobre sesgos de datos y equidad de las variables (como `gender` o `SeniorCitizen`), consulta `docs/ETHICS.md`.

> **Licencia:** Distribuido bajo la Licencia MIT.
