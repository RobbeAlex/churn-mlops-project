# ✅ Calidad de Código y Pruebas

Para un proyecto MLOps estableciendo prácticas de software, la calidad de código (linting, formateo) y la automatización de pruebas (CI) son esenciales.

## 1. Mantenimiento de Estilo con `pre-commit`

Para evitar subir código desordenado o no estándar a la rama principal, el repositorio utiliza ganchos (hooks) de **`pre-commit`**. 

El archivo `.pre-commit-config.yaml` define las siguientes herramientas que se ejecutan automáticamente cada vez que realizas un `git commit`:

- **Black**: Formateador de código de Python (PEP 8 estricto).
- **Flake8**: Linter para encontrar posibles errores de sintaxis y código muerto.
- **isort**: Ordena las importaciones de las librerías alfabéticamente y por bloques de forma limpia.
- **Trailing-whitespace & End-of-file-fixer**: Limpia espacios en blanco innecesarios y asegura que los archivos terminen correctamente.

**Cómo asegurar su ejecución:**
1. Instalar los hooks localmente (solo una vez tras clonar):
   ```bash
   pre-commit install
   ```
2. Al hacer commit, el hook evaluará los archivos modificados. Si algo falla o se formatea, el commit se detiene. Debes volver a añadir los archivos (`git add .`) y hacer commit de nuevo.

## 2. Pruebas Automatizadas con `pytest`

Las pruebas unitarias y de integración aseguran que el pipeline no se rompa al agregar nuevas funciones o al cambiar dependencias.

- **Carpeta:** Todas las pruebas están ubicadas en el directorio `test/`.
- **Script principal:** `test_pipeline.py`.
- **Herramienta:** `pytest`.

### ¿Qué probamos?
- **Validación de Carga de Datos:** Se comprueba que `data_loader.py` descargue y manipule las columnas correctamente, y que la dimensión de los datos procesados coincida con la entrada.
- **Consistencia del Modelo:** Se verifican funciones utilitarias en el pipeline.
- **La API (`test_predict_endpoint`):** A través del `TestClient` de FastAPI, enviamos un *request* falso al endpoint `/predict` y evaluamos que responda con un HTTP 200 y que devuelva el esquema correcto (`prediction`, `label`, `probability`).

**Ejecutar pruebas localmente:**
```bash
pytest test/
```

## 3. Cobertura de Código (Codecov)

Para medir qué porcentaje de nuestro código fuente está siendo ejecutado y validado por los tests (Unit Coverage), integramos **Codecov**.

1. En el pipeline de Integración Continua (GitHub Actions, `.github/workflows/`), se ejecutan las pruebas de pytest generando un archivo XML con la cobertura.
   ```bash
   pytest --cov=src --cov-report=xml
   ```
2. Este archivo `.coverage` se carga automáticamente a Codecov mediante la acción respectiva.
3. Puedes ver la medalla (badge) de cobertura en el archivo `README.md` principal, y asegurarte de mantenerla lo más alta posible.
