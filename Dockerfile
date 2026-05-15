# 1. Imagen base ligera de Python
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Evitar que Python genere archivos .pyc y permitir logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# CRITICAL: Configurar PYTHONPATH para que reconozca el módulo 'src'
ENV PYTHONPATH=/app

# 4. Instalar dependencias del sistema necesarias (si las hubiera)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copiar solo el archivo de requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar
# Asegúrate de haber limpiado el .dockerignore antes de este paso
COPY . .

# 7. Exponer el puerto de FastAPI
EXPOSE 8000

# 8. Comando por defecto (Lanzar la API)
# Usamos el formato de módulo para que uvicorn encuentre src/api.py
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]