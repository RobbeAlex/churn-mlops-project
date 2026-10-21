# Documentación Técnica: Churn MLOps Enterprise Pipeline
**Autor:** Roberto Alejandro Chagra Martínez  
**Tecnologías:** DVC, MLflow, FastAPI, Scikit-Learn, PyTest, Joblib, YAML, Docker/Conda  
**Tipo:** Pipeline de Producción MLOps con Versionado, Trazabilidad y API REST  

---

## 1. Descripción del Proyecto
Proyecto integral de MLOps para la predicción de deserción de clientes (Churn) en telecomunicaciones (*Telco-Customer-Churn*). Diseñado bajo estándares de ingeniería de software para aprendizaje automático: reproducibilidad de datos mediante **DVC**, seguimiento de experimentos con **MLflow**, configuración centralizada por archivos **YAML**, servicio desacoplado en **FastAPI** y suite de pruebas unitarias continuas con **PyTest**.

```mermaid
flowchart TD
    subgraph Data Pipeline (DVC)
        A["data/raw/Telco-Customer-Churn.csv"] --> B["src/data_loader.py"]
        B --> C["data/processed/train.csv
data/processed/test.csv"]
    end
    
    subgraph Experimentation & Training (MLflow)
        C --> D["src/model_trainer.py"]
        Config["config/params.yaml"] --> D
        D --> MLflow["MLflow Server (Tracking: F1, Recall, Acc)"]
        D --> E["models/model.joblib"]
    end
    
    subgraph Serving & Continuous Testing
        E --> API["src/api.py (FastAPI Async Lifespan)"]
        API --> Tests["test/test_pipeline.py (PyTest Client)"]
        API --> Client["Clientes / BI Dashboards (/predict)"]
    end
```

---

## 2. Estructura del Proyecto
```
churn-mlops-project/
├── config/
│   └── params.yaml            # Configuración de hiperparámetros y rutas
├── data/
│   ├── raw/                   # Datos crudos versionados
│   └── processed/             # Conjuntos train/test preprocesados
├── models/
│   └── model.joblib           # Artefacto binario serializado
├── scripts/
│   ├── download_dataset.py    # Ingesta automatizada desde Kaggle
│   └── generate_plots.py      # Curvas de ROC, AUC y matrices de confusión
├── src/
│   ├── api.py                 # Servicio REST de alta disponibilidad (FastAPI)
│   ├── data_loader.py         # Pipeline de limpieza y partición
│   ├── model_trainer.py       # Entrenamiento con registro en MLflow
│   └── predict.py             # Script de inferencia por lotes
└── test/
    └── test_pipeline.py       # Pruebas automatizadas de contrato y rendimiento
```

---

## 3. Especificación de la API de Inferencia ([`src/api.py`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/churn-mlops-project/src/api.py))
* **Gestor de Ciclo de Vida:** Utiliza `@asynccontextmanager lifespan(app)` para carga atómica del modelo en memoria durante el inicio del servidor y liberación segura al detenerse.
* **Endpoint de Inferencia:** `POST /predict`
* **Contrato de Entrada (Pydantic v2):**
  ```python
  class ChurnInput(BaseModel):
      gender: str
      SeniorCitizen: int
      Partner: int
      Dependents: int
      tenure: int
      PhoneService: int
      MultipleLines: str
      # ... contrato de 19 variables demográficas y de servicio
  ```
* **Respuesta JSON:**
  ```json
  {
      "churn_prediction": 1,
      "churn_probability": 0.784,
      "risk_level": "Alto"
  }
  ```

---

## 4. Guía de Ejecución
1. **Entrenar y registrar en MLflow:**
   ```bash
   python src/model_trainer.py
   ```
2. **Ejecutar suite de pruebas unitarias:**
   ```bash
   pytest test/test_pipeline.py -v
   ```
3. **Levantar el microservicio de inferencia:**
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
   ```
