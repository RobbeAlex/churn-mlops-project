# 🚀 Guía de Instalación y Uso

Este documento te guiará para configurar tu entorno local y levantar el proyecto, ya sea para desarrollo, reentrenamiento del modelo, o para consumir la API.

Tienes dos opciones principales: Instalación local usando entornos virtuales de Python (recomendado para desarrollo activo) o Instalación con Docker (ideal para producción y pruebas sin complicaciones de dependencias).

---

## Opción A: Instalación Local (Virtual Environment)

Esta es la mejor opción si planeas modificar el código, agregar nuevas features o ejecutar pruebas de forma nativa.

### 1. Clonar el repositorio

Abre tu terminal y ejecuta:

```bash
git clone https://github.com/RobbeAlex/churn-mlops-project.git
cd churn-mlops-project
```

### 2. Crear y activar el entorno virtual

Crearemos un entorno virtual para aislar las librerías necesarias de tu sistema.

```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Linux / macOS:
source .venv/bin/activate

# Activar en Windows (PowerShell):
.venv\Scripts\activate
```

### 3. Instalar dependencias

Con el entorno activado, instala todas las dependencias listadas en el `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Configurar Hooks (Desarrolladores)

Si vas a contribuir con código, es **obligatorio** configurar `pre-commit` para garantizar la calidad del código.

```bash
pre-commit install
```

---

## Opción B: Instalación con Docker (Alternativa Recomendada)

Si solo deseas correr la API o no quieres lidiar con configuraciones de Python, utiliza Docker.

```bash
git clone https://github.com/RobbeAlex/churn-mlops-project.git
cd churn-mlops-project

# Construir la imagen de Docker
docker-compose build
```

---

## 🏃 Cómo usar el proyecto

### 1. Reproducir el Pipeline de Entrenamiento

Gracias a la integración con **DVC** (Data Version Control), puedes ejecutar el flujo completo (cargar datos, limpiar y entrenar el modelo) con un solo comando. DVC solo ejecutará los pasos cuyos datos o código subyacente hayan cambiado.

```bash
dvc repro
```

> **Ver resultados con MLflow:** Si quieres analizar las métricas registradas, corre en tu terminal: `mlflow ui`. Luego abre `http://127.0.0.1:5000` en tu navegador.

### 2. Levantar la API de Inferencia

Una vez el modelo está entrenado, puedes levantar la API web.

**Si estás usando el entorno virtual local:**
```bash
uvicorn src.api:app --reload
```
*(El servidor estará escuchando en http://127.0.0.1:8000)*

**Si estás usando Docker Compose:**
```bash
docker-compose up -d api
```

### 3. Probar la API (Swagger UI)

FastAPI genera documentación interactiva automáticamente.

Abre tu navegador y ve a:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Desde allí, puedes ver el esquema de datos requerido y hacer peticiones de prueba utilizando el botón "Try it out". En la página de [Despliegue de la API](./Despliegue-de-la-API) encontrarás ejemplos detallados en `curl`.
