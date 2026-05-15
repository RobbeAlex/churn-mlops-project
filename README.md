# 📡 Proyecto Colaborativo MLOps: Predicción de Churn

## 🛠️ Tecnologías y Lenguajes

- **Python** (94.3%): Scripts, lógica del pipeline, API y procesamiento principal.
- **Dockerfile** (5.7%): Contenedores y despliegue para portabilidad y reproducibilidad.

## ¿Qué hace este proyecto?

Pipeline de Machine Learning modular y reproducible para predecir si un cliente de telecomunicaciones abandonará el servicio (**Churn**). El proyecto simula un entorno laboral real donde **4 roles colaboran** bajo buenas prácticas de MLOps.

Incluye preprocesamiento de datos, entrenamiento de modelos (Random Forest / Regresión Logística), serialización del modelo entrenado y una API REST lista para producción con FastAPI.

---

## 🏆 Resultados del Mejor Modelo

Algoritmo seleccionado: `RandomForestClassifier` con configuración definida en `config/params.yaml` (100 estimadores, profundidad máxima de 10).

| Métrica                | Valor  |
|------------------------|--------|
| Accuracy (Exactitud)   | 0.8155 |
| Recall (Exhaustividad) | 0.5389 |
| F1 Score               | 0.6073 |

---

## ¿Qué dataset usa y para qué sirve?

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-custom)  
**Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
**Problema:** Clasificación Binaria — ¿El cliente se va? (`Yes` / `No`)

Contiene 7,043 registros de clientes con variables demográficas, servicios contratados (teléfono, internet, seguridad, etc.) y datos financieros. Se usa para entrenar un clasificador binario que predice la baja de clientes.

> **Instrucción importante:**
> 1. Descarga el CSV desde el enlace de arriba.
> 2. Guárdalo en `data/raw/`.
> 3. **No subas el CSV a Git** (ya está en `.gitignore`). Cada miembro del equipo debe descargarlo localmente.

---

## 📂 Estructura de Carpetas

```text
churn-mlops-project/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── DATASET.md
├── ETHICS.md
├── README.md
├── Tarea__Proyecto_Telco_Customer_Churn.pdf
├── config/
│   └── params.yaml
├── data/
│   └── raw/
├── download_dataset.py
├── docker-compose.yml
├── models/
├── requirements.txt
├── src/
│   ├── api.py
│   ├── data_loader.py
│   ├── main.py
│   ├── model_trainer.py
│   └── predict.py
└── test/
    └── test_pipeline.py
```

---

## ¿Cómo lo instalo?

**Requisitos:** Python 3.9+

```bash
# 1. Clonar el repositorio
git clone https://github.com/RobbeAlex/churn-mlops-project
cd churn-mlops-project

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```
---

## Descripción breve de los scripts adicionales

- **download_dataset.py**: Descarga automática del dataset de churn si el archivo no existe en `data/raw/`.
- **Dockerfile**: Configura todo el entorno (dependencias y scripts) en una imagen de Docker.
- **docker-compose.yml**: Orquesta los servicios de entrenamiento (`train`) y despliegue de la API (`api`).
- **config/params.yaml**: Control centralizado de hiperparámetros y división entre train/test.

---

## 👥 Roles y Responsabilidades

1. **Data Engineer (`src/data_loader.py`):** Limpieza de datos (imputación de valores nulos en `TotalCharges`), eliminación de identificadores y transformación de variables categóricas.
2. **ML Engineer (`src/model_trainer.py`):** Implementación de la fábrica de modelos (Random Forest y Regresión Logística), cálculo de métricas y serialización del modelo.
3. **MLOps Engineer (`src/main.py` y `config/`):** Orquestación del flujo completo, gestión de hiperparámetros en `params.yaml` y configuración de rutas dinámicas absolutas.
4. **QA & Production Engineer (`src/predict.py` y `tests/`):** Validación mediante pruebas unitarias (`pytest`) y preparación del script de inferencia con manejo de errores.

---

## 🚀 Flujo de Trabajo con Git

1. **Clonar:** `git clone https://github.com/RobbeAlex/churn-mlops-project`
2. **Ramas:** Cada alumno crea su rama:
   - `git checkout -b feature/data-engineer`
   - `git checkout -b feature/ml-engineer`
   - `git checkout -b feature/mlops-engineer`
   - `git checkout -b feature/qa-engineer`
3. **Desarrollo:** Trabajen en paralelo con commits frecuentes.
4. **Integración:** El **MLOps Engineer** crea un Pull Request integrando todas las ramas a `main`. Resuelvan conflictos juntos si dos personas tocaron el mismo archivo.
5. **Prueba final:** Ejecuten `python -m src.main` en `main`. Si corre, ¡misión cumplida!

---

## ¿Cómo lo ejecuto?

### 1. **Clonar el repositorio**:

```bash
   `git clone https://github.com/RobbeAlex/churn-mlops-project`
```

Carga los datos, entrena el modelo configurado en `config/params.yaml` y guarda el `.pkl` en `models/`.


### 2. **Entrenar el modelo**:

```bash
    `docker-compose run train`
```

### 3. Lanzar la API

```bash
    `docker-compose up -d api`
```

Disponible en `http://127.0.0.1:8000`. Documentación interactiva en `http://127.0.0.1:8000/docs`.

### 4. **Probar la API**:

```bash
    Acceder a `http://localhost:8000/docs` para realizar predicciones.
```

---

## Ejemplo de llamada a la API

### Con `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "tenure": 2,
           "MonthlyCharges": 70.5,
           "TotalCharges": 141.0,
           "gender": 1,
           "Partner": 0
         }'
```

### Con `requests` (Python)

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "tenure": 2,
    "MonthlyCharges": 70.5,
    "TotalCharges": 141.0,
    "gender": 1,   # 1 = Female, 0 = Male
    "Partner": 0   # 1 = Yes, 0 = No
}

response = requests.post(url, json=payload)
print(response.json())
# {"prediction": 1, "label": "Churn", "probability": 0.73}
```

---

## 🤖 Contribución de IA

Se utilizó IA (Gemini) como apoyo técnico para resolver desafíos de arquitectura e integración:

- **Data Engineer:** Optimización de la limpieza de `TotalCharges` y estructuración de la división del dataset según `params.yaml`.
- **ML Engineer:** Estructuración de la función de entrenamiento, métricas de validación y serialización con `joblib`.
- **MLOps Engineer:** Depuración de errores de importación (`ModuleNotFoundError`) y rutas absolutas dinámicas con `os`.
- **QA Engineer:** Redacción de casos de prueba en `test_pipeline.py` y manejo de excepciones en `predict.py`.

---

## ✅ Checklist de Entrega

- [ ] `python -m src.main` ejecuta todo el pipeline sin errores.
- [ ] `config/params.yaml` existe y controla los hiperparámetros.
- [ ] Hay al menos 2 modelos implementados en el código.
- [ ] `predict.py` carga el modelo y realiza una predicción de ejemplo.
- [ ] El historial de Git muestra contribuciones de los 4 miembros.
- [ ] El `README.md` incluye los resultados obtenidos (Accuracy / Recall del mejor modelo)

