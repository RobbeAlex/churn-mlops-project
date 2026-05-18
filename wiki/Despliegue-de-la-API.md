# 🌐 Despliegue de la API

Para que el modelo de Machine Learning sea útil para la compañía de telecomunicaciones, debe integrarse con otros sistemas. Para ello, hemos expuesto el modelo entrenado como un servicio web a través de una **API REST**, empaquetada usando **Docker**.

## Tecnologías Utilizadas
- **FastAPI:** Framework web moderno y rápido para construir APIs en Python.
- **Uvicorn:** Servidor ASGI de alto rendimiento.
- **Pydantic:** Validación de los tipos de datos en la entrada (request).
- **joblib:** Para la carga del modelo binario serializado.

## Archivo Principal: `src/api.py`

En este script, FastAPI crea el servidor web.
1. Al arrancar (`@app.on_event("startup")`), se carga el modelo en memoria desde la ruta `models/model.joblib`.
2. Se define un esquema de datos `CustomerData` heredando de `BaseModel` (Pydantic). Este esquema asegura que todas las peticiones `POST` incluyan las 19 variables (features) correctas requeridas por el modelo.
3. El endpoint principal `/predict` recibe los datos, los convierte en un array compatible con scikit-learn, ejecuta la predicción `model.predict()` y calcula las probabilidades `model.predict_proba()`.

---

## Documentación y Prueba Interactiva

Una gran ventaja de FastAPI es que genera documentación OpenAPI (Swagger) automáticamente.
Una vez levantado el servidor, entra a tu navegador:

🔗 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

Desde ahí, puedes visualizar los endpoints y utilizar la funcionalidad "Try it out" para enviar peticiones directamente desde la interfaz gráfica.

---

## Endpoints Disponibles

### 1. Health Check
Endpoint para verificar que el servicio está activo.
- **Método:** `GET`
- **Ruta:** `/`
- **Respuesta Exitosa:**
```json
{
  "message": "Welcome to the Churn Prediction API"
}
```

### 2. Predicción de Churn
El endpoint principal que procesa la inferencia del modelo.
- **Método:** `POST`
- **Ruta:** `/predict`
- **Cuerpo de la Petición (Request Body):** JSON con 19 características categóricas y numéricas del cliente (ver `CustomerData` en el código fuente para los detalles y tipos).

**Ejemplo de Petición con `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "tenure": 2, "MonthlyCharges": 70.5, "TotalCharges": 141.0, 
           "gender": 1, "Partner": 0, "Dependents": 0, "PhoneService": 1, 
           "MultipleLines": 0, "InternetService": 2, "OnlineSecurity": 0, 
           "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0, 
           "StreamingTV": 0, "StreamingMovies": 0, "Contract": 0, 
           "PaperlessBilling": 1, "PaymentMethod": 2, "SeniorCitizen": 0
         }'
```

**Ejemplo de Respuesta Exitosa:**
```json
{
  "prediction": 1,
  "label": "Churn",
  "probability": 0.73
}
```

- `prediction`: Clase numérica retornada por el modelo (0 o 1).
- `label`: Etiqueta legible ('Churn' o 'No Churn').
- `probability`: Grado de certeza del modelo de pertenecer a esa clase.

---

## Empaquetado con Docker

Para asegurar que la API corra idénticamente en cualquier máquina de desarrollo o servidor en la nube, la API está *dockerizada*.

- **`Dockerfile`:** Define una imagen base de Python 3.9 delgada (`slim`), instala los requerimientos, copia el código y expone el puerto 8000.
- **`docker-compose.yml`:** Orquesta la construcción y despliegue del contenedor con un comando simplificado.

**Para ejecutar vía Docker:**
```bash
docker-compose up -d api
```
*(El servicio estará disponible en el puerto 8000 de tu localhost).*
