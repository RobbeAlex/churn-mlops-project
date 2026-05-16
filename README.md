# 📡 Proyecto MLOps: Predicción de Churn en Telecomunicaciones

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Python application](https://github.com/RobbeAlex/churn-mlops-project/actions/workflows/python-app.yml/badge.svg)](https://github.com/RobbeAlex/churn-mlops-project/actions/workflows/python-app.yml)

Pipeline de Machine Learning modular y reproducible para predecir si un cliente de telecomunicaciones abandonará el servicio (**Churn**). El proyecto simula un entorno laboral real donde **4 roles colaboran** bajo buenas prácticas de MLOps, desde la ingesta de datos hasta el despliegue de una API REST lista para producción.

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
- [Testing](#-testing)
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
- **API REST con FastAPI** lista para producción
- **Contenerización con Docker** para máxima reproducibilidad
- **Pipeline orquestado** mediante configuración centralizada en YAML
- **Tests unitarios** con pytest

El objetivo es predecir si un cliente de telecomunicaciones cancelará su servicio en el próximo ciclo, permitiendo a la empresa tomar acciones proactivas de retención.

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
| **API Framework** | FastAPI | 0.136.1 | Servidor web para predicciones en tiempo real |
| **Servidor ASGI** | Uvicorn | 0.47.0 | Servidor de aplicación para FastAPI |
| **Data Processing** | pandas | 2.2.2 | Manipulación y limpieza de datos |
| **Data Processing** | numpy | 1.26.4 | Operaciones numéricas |
| **Serialización** | joblib | 1.5.3 | Guardado/carga del modelo entrenado |
| **Configuración** | PyYAML | 6.0.3 | Gestión de hiperparámetros |
| **Testing** | pytest | 9.0.3 | Framework de pruebas unitarias |
| **Validación** | Pydantic | 2.13.4 | Validación de esquemas de entrada API |
| **Contenerización** | Docker | - | Empaquetado y despliegue reproducible |
| **Orquestación** | Docker Compose | - | Gestión multi-contenedor |

---

## 📦 Prerrequisitos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.9 o superior** ([Descargar](https://www.python.org/downloads/))
- **Git** ([Descargar](https://git-scm.com/downloads))
- **Docker Desktop** (opcional, pero recomendado) ([Descargar](https://www.docker.com/products/docker-desktop/))
  - Docker Engine >= 20.10
  - Docker Compose >= 2.0
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

# En Linux/macOS:
source .venv/bin/activate

# En Windows:
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset
python download_dataset.py
# El archivo CSV se guardará en: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Opción B: Instalación con Docker (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/RobbeAlex/churn-mlops-project.git
cd churn-mlops-project

# 2. Construir la imagen Docker (se hace automáticamente con docker-compose)
docker-compose build

# ¡Listo! No necesitas instalar Python ni dependencias localmente
```

---

## 📊 Dataset

**Nombre:** Telco Customer Churn  
**Fuente:** [Kaggle — IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Descripción

- **Registros:** 7,043 clientes
- **Variables:** 21 columnas (demográficas, servicios contratados, facturación)
- **Problema:** Clasificación binaria — ¿El cliente cancelará? (`Yes` / `No`)
- **Distribución:** ~26% Churn / ~74% No Churn (desbalanceado)

### Variables principales

- **Demográficas:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Servicios:** `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`
- **Financieras:** `tenure` (meses), `MonthlyCharges`, `TotalCharges`
- **Contractuales:** `Contract` (Month-to-month, One year, Two year), `PaymentMethod`
- **Objetivo:** `Churn` (Yes/No)

### ⚠️ Descarga del Dataset

**El dataset NO está incluido en el repositorio por razones de tamaño.**

**Opción 1 — Descarga automática (recomendada):**
```bash
python download_dataset.py
```

**Opción 2 — Descarga manual:**
1. Visita: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Descarga `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Guárdalo en: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

> 📝 **Nota:** El archivo CSV está incluido en `.gitignore` para no subir datos crudos a control de versiones.

Para más detalles sobre el dataset, consulta [DATASET.md](DATASET.md).

---

## 📂 Estructura del Proyecto

```
churn-mlops-project/
│
├── .github/
│   └── workflows/              # CI/CD: tests automáticos con GitHub Actions
│
├── config/
│   └── params.yaml             # ⚙️ Hiperparámetros centralizados
│
├── data/
│   └── raw/                    # 📁 Carpeta para el dataset CSV (no en Git)
│
├── models/                     # 🤖 Modelos entrenados (.pkl) — generados en runtime
│
├── src/
│   ├── api.py                  # 🌐 API REST con FastAPI
│   ├── data_loader.py          # 🔧 ETL: carga, limpieza y transformación
│   ├── main.py                 # 🎯 Orquestador principal del pipeline
│   ├── model_trainer.py        # 🧠 Fábrica de modelos ML
│   └── predict.py              # 🔮 Script de inferencia
│
├── test/
│   └── test_pipeline.py        # ✅ Tests unitarios con pytest
│
├── tmp/
│   └── pytest-of-codespace/    # Archivos temporales de pytest
│
├── .dockerignore               # Exclusiones para imagen Docker
├── .gitignore                  # Exclusiones para Git
├── DATASET.md                  # 📊 Documentación completa del dataset
├── ETHICS.md                   # ⚖️ Análisis ético y sesgos del modelo
├── README.md                   # 📖 Este archivo
├── Dockerfile                  # 🐳 Imagen Docker del proyecto
├── docker-compose.yml          # 🎼 Orquestación de servicios
├── download_dataset.py         # ⬇️ Script de descarga automática
├── requirements.txt            # 📦 Dependencias Python con versiones fijas
└── Tarea__Proyecto_Telco_Customer_Churn.pdf  # 📄 Enunciado original del proyecto
```

---

## 🏃 Cómo Ejecutar

### Flujo Completo del Pipeline

```
download_dataset.py
       ↓
data/raw/*.csv
       ↓
src/data_loader.py  ──→  Limpieza + Encoding + Split train/test
       ↓
src/model_trainer.py ──→  Entrena RandomForest (según params.yaml)
       ↓                   Calcula Accuracy, Recall, F1-Score
models/model.pkl
       ↓
src/api.py          ──→  API en :8000 → POST /predict
```

### Ejecución Local (Sin Docker)

```bash
# 1. Asegurarse de que el dataset esté descargado
python download_dataset.py

# 2. Entrenar el modelo completo
python -m src.main
# Salida esperada:
# Loading data from data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv...
# Data loaded successfully: 7043 rows, 21 columns
# Training model: RandomForest...
# Model trained successfully!
# Accuracy: 0.8155
# Recall: 0.5389
# F1-Score: 0.6073
# Model saved to: models/model.pkl

# 3. Ejecutar tests unitarios
python -m pytest test/ -v
# Todos los tests deben pasar en verde ✅

# 4. Levantar la API en modo desarrollo
uvicorn src.api:app --reload
# API corriendo en: http://127.0.0.1:8000
# Documentación interactiva: http://127.0.0.1:8000/docs
```

### Ejecución con Docker (Recomendado)

```bash
# 1. Entrenar el modelo (descarga dataset + entrena)
docker-compose run train
# Este comando ejecuta el pipeline completo dentro de un contenedor

# 2. Levantar la API en modo producción
docker-compose up -d api
# API corriendo en: http://localhost:8000
# Ver logs: docker-compose logs -f api

# 3. Detener la API
docker-compose down
```

---

## 🌐 Uso de la API

### Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Mensaje de bienvenida |
| GET | `/health` | Health check del servicio |
| POST | `/predict` | Realiza una predicción de churn |

### Documentación Interactiva

Una vez que la API esté corriendo, visita:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

### Esquema de Entrada (POST /predict)

```json
{
  "tenure": 2,
  "MonthlyCharges": 70.5,
  "TotalCharges": 141.0,
  "gender": 1,
  "Partner": 0,
  "Dependents": 0,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 2,
  "OnlineSecurity": 0,
  "OnlineBackup": 0,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 0,
  "StreamingMovies": 0,
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": 2,
  "SeniorCitizen": 0
}
```

> **Nota sobre encoding:** Las variables categóricas ya deben estar codificadas numéricamente. Por ejemplo:
> - `gender`: 0 = Male, 1 = Female
> - `InternetService`: 0 = DSL, 1 = Fiber optic, 2 = No
> - `Contract`: 0 = Month-to-month, 1 = One year, 2 = Two year

### Esquema de Respuesta

```json
{
  "prediction": 1,
  "label": "Churn",
  "probability": 0.73
}
```

- `prediction`: 0 (No Churn) o 1 (Churn)
- `label`: Etiqueta textual ("No Churn" / "Churn")
- `probability`: Probabilidad de la clase predicha (0.0 - 1.0)

### Ejemplos de Llamadas

#### Con `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "tenure": 2,
           "MonthlyCharges": 70.5,
           "TotalCharges": 141.0,
           "gender": 1,
           "Partner": 0,
           "Dependents": 0,
           "PhoneService": 1,
           "MultipleLines": 0,
           "InternetService": 2,
           "OnlineSecurity": 0,
           "OnlineBackup": 0,
           "DeviceProtection": 0,
           "TechSupport": 0,
           "StreamingTV": 0,
           "StreamingMovies": 0,
           "Contract": 0,
           "PaperlessBilling": 1,
           "PaymentMethod": 2,
           "SeniorCitizen": 0
         }'
```

#### Con Python `requests`

```python
import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "tenure": 2,
    "MonthlyCharges": 70.5,
    "TotalCharges": 141.0,
    "gender": 1,           # 1 = Female
    "Partner": 0,          # 0 = No
    "Dependents": 0,       # 0 = No
    "PhoneService": 1,     # 1 = Yes
    "MultipleLines": 0,    # 0 = No
    "InternetService": 2,  # 2 = Fiber optic
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 0,         # 0 = Month-to-month
    "PaperlessBilling": 1, # 1 = Yes
    "PaymentMethod": 2,    # 2 = Electronic check
    "SeniorCitizen": 0     # 0 = No
}

response = requests.post(url, json=payload)
print(response.json())
# Salida: {"prediction": 1, "label": "Churn", "probability": 0.73}
```

#### Con JavaScript `fetch`

```javascript
const url = "http://127.0.0.1:8000/predict";

const payload = {
    tenure: 2,
    MonthlyCharges: 70.5,
    TotalCharges: 141.0,
    gender: 1,
    Partner: 0,
    Dependents: 0,
    PhoneService: 1,
    MultipleLines: 0,
    InternetService: 2,
    OnlineSecurity: 0,
    OnlineBackup: 0,
    DeviceProtection: 0,
    TechSupport: 0,
    StreamingTV: 0,
    StreamingMovies: 0,
    Contract: 0,
    PaperlessBilling: 1,
    PaymentMethod: 2,
    SeniorCitizen: 0
};

fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
})
.then(response => response.json())
.then(data => console.log(data));
// Salida: {prediction: 1, label: "Churn", probability: 0.73}
```

---

## 👥 Roles y Responsabilidades

Este proyecto simula un equipo MLOps real con 4 roles especializados:

| Rol | Archivos principales | Responsabilidades |
|-----|----------------------|-------------------|
| **Data Engineer** | `src/data_loader.py` | Limpieza de datos (imputación de `TotalCharges`), eliminación de identificadores, transformación de variables categóricas, división train/test |
| **ML Engineer** | `src/model_trainer.py` | Implementación de modelos (Random Forest, Regresión Logística), cálculo de métricas (Accuracy, Recall, F1), serialización del modelo con joblib |
| **MLOps Engineer** | `src/main.py`, `config/params.yaml` | Orquestación del pipeline completo, gestión de hiperparámetros centralizados, configuración de rutas y parámetros |
| **QA & Production Engineer** | `src/predict.py`, `test/test_pipeline.py`, `src/api.py` | Validación mediante tests unitarios, preparación del script de inferencia, despliegue de la API con FastAPI, manejo de errores |

---

## 🔀 Flujo de Trabajo con Git

### Estrategia de Branching

Este proyecto utiliza una estrategia de ramas por funcionalidad/rol:

```
main (rama principal — siempre estable)
 ├── feature/data-engineer
 ├── feature/ml-engineer
 ├── feature/mlops-engineer
 └── feature/qa-engineer
```

### Proceso de Colaboración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/RobbeAlex/churn-mlops-project.git
   cd churn-mlops-project
   ```

2. **Crear rama personal según tu rol:**
   ```bash
   # Ejemplo para Data Engineer:
   git checkout -b feature/data-engineer
   
   # Otros ejemplos:
   # git checkout -b feature/ml-engineer
   # git checkout -b feature/mlops-engineer
   # git checkout -b feature/qa-engineer
   ```

3. **Desarrollar con commits frecuentes:**
   ```bash
   git add .
   git commit -m "feat(data): implementar limpieza de TotalCharges"
   git push origin feature/data-engineer
   ```

4. **Integración mediante Pull Request:**
   - El **MLOps Engineer** coordina la integración de todas las ramas
   - Se crea un Pull Request de cada rama hacia `main`
   - El equipo revisa el código en conjunto
   - Se resuelven conflictos si dos personas modificaron el mismo archivo
   - Una vez aprobado, se hace merge a `main`

5. **Validación final:**
   ```bash
   git checkout main
   git pull origin main
   python -m src.main  # Debe ejecutarse sin errores
   ```

### Convenciones de Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): descripción` — Nueva funcionalidad
- `fix(scope): descripción` — Corrección de bug
- `docs(scope): descripción` — Cambios en documentación
- `test(scope): descripción` — Añadir o modificar tests
- `refactor(scope): descripción` — Refactorización de código

**Ejemplos:**
```bash
git commit -m "feat(model): añadir soporte para Regresión Logística"
git commit -m "fix(api): corregir encoding de variables en endpoint /predict"
git commit -m "docs(readme): actualizar instrucciones de instalación"
git commit -m "test(data): añadir tests para validación de CSV"
```

---

## ✅ Testing

### Ejecutar Tests

```bash
# Ejecutar todos los tests
python -m pytest test/

# Ejecutar con verbose (más detalles)
python -m pytest test/ -v

# Ejecutar con cobertura de código
python -m pytest test/ --cov=src --cov-report=html
```

### Tests Implementados

| Test | Archivo | Descripción |
|------|---------|-------------|
| `test_data_loading` | `test/test_pipeline.py` | Verifica que el CSV se carga correctamente |
| `test_data_preprocessing` | `test/test_pipeline.py` | Valida que la limpieza de datos funciona |
| `test_model_training` | `test/test_pipeline.py` | Comprueba que el modelo entrena sin errores |
| `test_model_prediction` | `test/test_pipeline.py` | Valida que las predicciones tienen el formato correcto |

### Cobertura de Código

Tras ejecutar `pytest --cov`, se genera un reporte en `htmlcov/index.html` que muestra:
- Líneas cubiertas por tests
- Líneas sin cubrir
- Porcentaje de cobertura por archivo

**Objetivo de cobertura:** >80%

---

## ⚖️ Consideraciones Éticas

Este proyecto incluye un análisis exhaustivo de sesgos, equidad y transparencia algorítmica en el archivo [ETHICS.md](ETHICS.md).

### Resumen de Hallazgos Clave

**Sesgos Identificados:**
- **Subrepresentación etaria:** Solo 16% de adultos mayores en el dataset
- **Género binario:** Excluye identidades no binarias
- **Proxy socioeconómico:** El método de pago correlaciona con nivel de ingresos

**Variables Sensibles:**
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Usar estas variables en decisiones automatizadas sin auditoría de equidad puede generar discriminación

**Limitaciones del Modelo:**
- **Recall bajo (0.54):** Pierde ~46% de clientes que sí abandonan
- **Desbalance de clases:** El modelo se sesga hacia predecir "No Churn"
- **Correlación ≠ Causalidad:** El modelo detecta patrones pero no entiende causas

**Recomendaciones:**
1. No usar `gender` ni `SeniorCitizen` en decisiones de precios o retención sin auditoría
2. Monitorear rendimiento del modelo por segmentos demográficos
3. Mantener supervisión humana en decisiones de alto impacto
4. Reentrenar periódicamente con datos actualizados

Para el análisis completo, consulta [ETHICS.md](ETHICS.md).

---

## 🔧 Solución de Problemas

### Error: `ModuleNotFoundError: No module named 'src'`

**Causa:** Python no encuentra el paquete `src`.

**Solución:**
```bash
# Siempre ejecutar desde la raíz del repositorio usando -m
python -m src.main

# NO ejecutar así:
# python src/main.py  ❌
```

### Error: `FileNotFoundError: data/raw/... not found`

**Causa:** El dataset no ha sido descargado.

**Solución:**
```bash
# Descargar automáticamente:
python download_dataset.py

# O descargar manualmente desde:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# y guardar en: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Error al correr Docker: `port already in use`

**Causa:** El puerto 8000 ya está ocupado por otro servicio.

**Solución 1 — Cambiar el puerto:**
```yaml
# Editar docker-compose.yml
services:
  api:
    ports:
      - "8001:8000"  # Cambiar 8000 por 8001
```

**Solución 2 — Detener el servicio que ocupa el puerto:**
```bash
# En Linux/macOS:
sudo lsof -i :8000
kill <PID>

# En Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Error: `permission denied` al ejecutar Docker

**Causa:** Usuario sin permisos para ejecutar Docker.

**Solución (Linux):**
```bash
# Añadir usuario al grupo docker
sudo usermod -aG docker $USER

# Cerrar sesión y volver a entrar
# O ejecutar con sudo:
sudo docker-compose run train
```

### Error: `ImportError: cannot import name 'joblib'`

**Causa:** Dependencias no instaladas correctamente.

**Solución:**
```bash
# Reinstalar dependencias
pip install --force-reinstall -r requirements.txt

# Si persiste, actualizar pip:
pip install --upgrade pip
pip install -r requirements.txt
```

### El modelo tiene baja precisión en mis datos

**Causa posible:** Tus datos son muy diferentes al dataset de entrenamiento (IBM Telco).

**Solución:**
1. Reentrenar el modelo con tus propios datos
2. Ajustar hiperparámetros en `config/params.yaml`
3. Considerar técnicas de balanceo de clases
4. Validar que el encoding de variables categóricas sea consistente

### Error: `yaml.scanner.ScannerError`

**Causa:** Archivo `config/params.yaml` con sintaxis incorrecta.

**Solución:**
- Validar que la indentación sea con espacios (no tabs)
- Usar un validador YAML online: https://www.yamllint.com/
- Revisar que no haya caracteres especiales sin escapar

---

## 🤖 Contribución de IA

Se utilizó IA (Gemini) como apoyo técnico para resolver desafíos de arquitectura e integración:

| Rol | Soporte de IA |
|-----|---------------|
| **Data Engineer** | Optimización de la limpieza de `TotalCharges` y estructuración de la división del dataset según `params.yaml` |
| **ML Engineer** | Estructuración de la función de entrenamiento, métricas de validación y serialización con `joblib` |
| **MLOps Engineer** | Depuración de errores de importación (`ModuleNotFoundError`) y rutas absolutas dinámicas con `os` |
| **QA Engineer** | Redacción de casos de prueba en `test_pipeline.py` y manejo de excepciones en `predict.py` |

---

## ✅ Checklist de Entrega

Antes de considerar el proyecto completo, verifica:

- [ ] `git clone` funciona sin errores
- [ ] `pip install -r requirements.txt` instala todas las dependencias
- [ ] `python download_dataset.py` descarga el CSV correctamente
- [ ] `python -m src.main` ejecuta el pipeline sin errores
- [ ] Se genera `models/model.pkl` tras el entrenamiento
- [ ] `python -m pytest test/` pasa todos los tests ✅
- [ ] `docker-compose run train` reproduce el entrenamiento
- [ ] `docker-compose up -d api` levanta la API en :8000
- [ ] El endpoint `POST /predict` responde correctamente
- [ ] `README.md`, `DATASET.md` y `ETHICS.md` están completos
- [ ] `.gitignore` excluye `data/`, `models/*.pkl`, `.venv/`

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 📞 Contacto

**Repositorio:** https://github.com/RobbeAlex/churn-mlops-project

Para reportar bugs o sugerir mejoras, abre un issue en GitHub.

---
