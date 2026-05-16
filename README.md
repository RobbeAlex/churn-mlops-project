# 📡 Proyecto MLOps: Predicción de Churn en Telecomunicaciones

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-purple)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Python application](https://github.com/RobbeAlex/churn-mlops-project/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/RobbeAlex/churn-mlops-project/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/RobbeAlex/churn-mlops-project/graph/badge.svg?token=WFGYQV4VB3)](https://codecov.io/gh/RobbeAlex/churn-mlops-project)

Pipeline de Machine Learning modular y reproducible para predecir si un cliente de telecomunicaciones abandonará el servicio (**Churn**). El proyecto simula un entorno laboral real donde **4 roles colaboran** bajo buenas prácticas de MLOps, desde la ingesta de datos hasta el despliegue de una API REST lista para producción. Incorpora además **DVC** para versionado de datos, **MLflow** para tracking de experimentos, y **Codecov** junto con **pre-commit** para asegurar la máxima calidad de código en Integración Continua (CI).

---

## 📋 Tabla de Contenidos

- [¿Qué hace este proyecto?](#-qué-hace-este-proyecto)
- [Resultados del Modelo](#-resultados-del-modelo)
- [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
- [Prerrequisitos](#-prerrequisitos)
- [Instalación](#-instalación)
- [Dataset](#-dataset)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Cómo Ejecutar](#-cómo-ejecutar)
- [Uso de la API](#-uso-de-la-api)
- [Roles y Responsabilidades](#-roles-y-responsabilidades)
- [Flujo de Trabajo con Git](#-flujo-de-trabajo-con-git)
- [Testing y Calidad de Código](#-testing-y-calidad-de-código)
- [Consideraciones Éticas](#-consideraciones-éticas)
- [Solución de Problemas](#-solución-de-problemas)
- [Contribución de IA](#-contribución-de-ia)
- [Licencia](#-licencia)

---

## 🎯 ¿Qué hace este proyecto?

Desarrolla un **sistema completo de predicción de abandono de clientes** (Customer Churn) que incluye:

- **Preprocesamiento automático** de datos (imputación, encoding, división train/test)
- **Entrenamiento de modelos** de Machine Learning (Random Forest / Regresión Logística)
- **Serialización del modelo** entrenado para reutilización
- **Tracking de Experimentos** usando MLflow para registro de métricas y parámetros
- **Control de Versiones de Datos (DVC)** para rastrear pipelines y datasets
- **Calidad de Código y Cobertura** usando pre-commit (Black, Ruff) y Codecov
- **API REST con FastAPI** lista para producción
- **Contenerización con Docker** para máxima reproducibilidad
- **Tests unitarios** automatizados mediante GitHub Actions

El objetivo es predecir si un cliente cancelará su servicio en el próximo ciclo, permitiendo tomar acciones de retención.

---

## 🏆 Resultados del Modelo

**Algoritmo seleccionado:** `RandomForestClassifier`  
**Configuración:** 100 estimadores, profundidad máxima de 10 (definido en `config/params.yaml`)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 0.8155 | 81.55% de predicciones correctas en general |
| **Recall** | 0.5389 | Detecta 53.89% de los clientes que realmente abandonan |
| **F1-Score** | 0.6073 | Balance entre precisión y exhaustividad |

> **⚠️ Nota importante:** El Recall de 0.54 indica que el modelo no detecta ~46% de los clientes que sí abandonan. Esto se debe al desbalance de clases en el dataset (74% No Churn / 26% Churn). Para aplicaciones críticas de negocio, considerar técnicas de balanceo como SMOTE o ajuste de `class_weight`.

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Lenguaje** | Python | 3.9+ | Lenguaje principal del proyecto |
| **ML Framework** | scikit-learn | 1.4.2 | Entrenamiento y evaluación de modelos |
| **Experiment Tracking** | MLflow | - | Seguimiento de parámetros, métricas y modelos |
| **Versionado de Datos** | DVC | - | Versionado de datasets y pipeline ML |
| **API Framework** | FastAPI | 0.136.1 | Servidor web para predicciones en tiempo real |
| **Servidor ASGI** | Uvicorn | 0.47.0 | Servidor de aplicación para FastAPI |
| **Data Processing** | pandas | 2.2.2 | Manipulación y limpieza de datos |
| **Testing** | pytest | 9.0.3 | Framework de pruebas unitarias |
| **Calidad de Código** | pre-commit | - | Hooks para linting (Ruff) y formateo (Black) |
| **Cobertura (CI)** | Codecov | - | Análisis de cobertura de código continuo |
| **Contenerización** | Docker | - | Empaquetado y despliegue reproducible |

---

## 📦 Prerrequisitos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.9 o superior**
- **Git**
- **Docker Desktop** (opcional, pero recomendado)
- **Cuenta en Kaggle** (para descarga automática del dataset) — Opcional si descargas manualmente

---

## 🚀 Instalación

### Opción A: Instalación Local (Sin Docker)

```bash
# 1. Clonar el repositorio
git clone https://github.com/RobbeAlex/churn-mlops-project.git
cd churn-mlops-project

# 2. Crear y activar entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar pre-commit (Calidad de código)
pre-commit install

# 5. Descargar el dataset (si no usas DVC)
python scripts/download_dataset.py
```

### Opción B: Instalación con Docker (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/RobbeAlex/churn-mlops-project.git
cd churn-mlops-project

# 2. Construir la imagen Docker
docker-compose build
```

---

## 📊 Dataset

**Nombre:** Telco Customer Churn  
**Fuente:** Kaggle — IBM Sample Data  
**Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Para más detalles sobre las variables y descarga, consulta [DATASET.md](DATASET.md).

---

## 📂 Estructura del Proyecto

```text
churn-mlops-project/
├── .github/workflows/          # CI/CD: pipelines en GitHub Actions (incluye Codecov)
├── .pre-commit-config.yaml     # Configuración de linting y formateo automático
├── codecov.yml                 # Configuración de cobertura de código
├── config/
│   └── params.yaml             # ⚙️ Hiperparámetros y settings centralizados
├── data/                       # 📁 Carpeta para datos raw y procesados (manejados por DVC)
├── docs/                       # Documentación adicional
├── dvc.yaml                    # 📈 Pipeline de Data Version Control (DVC)
├── models/                     # 🤖 Modelos entrenados (.pkl)
├── scripts/                    # Scripts auxiliares (e.g. descargar dataset)
├── src/
│   ├── api.py                  # 🌐 API REST con FastAPI
│   ├── data_loader.py          # 🔧 ETL: carga, limpieza y transformación
│   ├── main.py                 # 🎯 Orquestador principal del pipeline
│   ├── model_trainer.py        # 🧠 Entrenamiento y logging con MLflow
│   └── predict.py              # 🔮 Script de inferencia
├── test/
│   └── test_pipeline.py        # ✅ Tests unitarios
├── Dockerfile                  # 🐳 Configuración Docker
├── docker-compose.yml          # 🎼 Orquestación de contenedores
├── requirements.txt            # 📦 Dependencias Python
└── README.md                   # 📖 Este archivo
```

---

## 🏃 Cómo Ejecutar

### 1. Ejecutar el Pipeline de ML con DVC
En lugar de correr los scripts manualmente, utilizamos DVC para reproducir el pipeline respetando el grafo de dependencias:
```bash
# Reproducir el pipeline completo (preparación de datos + entrenamiento)
dvc repro
```
Esto asegurará que si cambian los datos o parámetros (`params.yaml`), se re-ejecutarán los pasos necesarios.

### 2. Levantar la Interfaz de MLflow
Puedes visualizar el historial de entrenamientos, métricas y experimentos registrados levantando la UI de MLflow:
```bash
mlflow ui
# Estará disponible en http://127.0.0.1:5000
```

### 3. Levantar la API
#### Usando Uvicorn (Local)
```bash
uvicorn src.api:app --reload
# API corriendo en: http://127.0.0.1:8000
```

#### Usando Docker Compose
```bash
docker-compose up -d api
```

---

## 🌐 Uso de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check del servicio |
| POST | `/predict` | Realiza una predicción de churn |

Puedes probar los endpoints usando la documentación interactiva en http://127.0.0.1:8000/docs.

---

## 👥 Roles y Responsabilidades

| Rol | Archivos principales | Responsabilidades |
|-----|----------------------|-------------------|
| **Data Engineer** | `src/data_loader.py` | Limpieza, encoding, split train/test y manejo de dependencias de datos en DVC |
| **ML Engineer** | `src/model_trainer.py` | Entrenamiento, métricas, y logging en MLflow de hiperparámetros |
| **MLOps Engineer** | `dvc.yaml`, `config/` | Orquestación, configuración centralizada y gestión de pipelines |
| **QA & Prod Engineer** | `test/`, `src/api.py`, `codecov.yml` | Pruebas unitarias, cobertura, pre-commit hooks y despliegue API |

---

## 🔀 Flujo de Trabajo con Git

Este proyecto usa una estrategia de ramas (`feature/rol`).
1. Crea tu rama: `git checkout -b feature/tu-rol`
2. **Importante:** Gracias a `pre-commit`, antes de cada commit el código se auto-formateará (Black) y se analizará en busca de errores (Ruff).
3. Haz commits: `git commit -m "feat: mi cambio"`
4. Abre un Pull Request a `main`.

---

## ✅ Testing y Calidad de Código

### Tests y Cobertura (Codecov)
El proyecto usa `pytest` y se integra con **Codecov** para asegurar una alta cobertura de código en cada PR.
```bash
# Ejecutar tests y generar reporte de cobertura
python -m pytest test/ --cov=src --cov-report=html
```

### Calidad Continua (pre-commit)
Para mantener el código limpio de manera uniforme usamos `black` y `ruff`. Se ejecutan solos en tus commits si instalaste los hooks:
```bash
pre-commit run --all-files
```

---

## ⚖️ Consideraciones Éticas

Consulta [ETHICS.md](ETHICS.md) para un análisis exhaustivo de sesgos, equidad y limitaciones del modelo (e.g. cuidado al usar `gender` o `SeniorCitizen`).

---

## 🔧 Solución de Problemas

- **DVC dice que todo está al día pero cambié código:** Recuerda que DVC usa caché. Usa `dvc repro -f` para forzar la ejecución.
- **Pre-commit falla en un commit:** Pre-commit corrige automáticamente algunos errores de formateo. Solo necesitas hacer `git add .` y volver a intentar el `git commit`.
- **Puerto 8000 ocupado por API:** Cambia el mapeo de puertos en `docker-compose.yml` a `8001:8000` o mata el proceso local.

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver archivo `LICENSE`.
