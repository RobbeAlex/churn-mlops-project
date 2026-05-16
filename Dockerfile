# ETAPA 1: Construcción (Builder)
# Utilizamos una imagen base ligera de Python
FROM python:3.11-slim as builder

# Configuramos variables de entorno para evitar que Python escriba archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos solo los requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalamos las dependencias en una carpeta específica
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ETAPA 2: Producción (Final)
FROM python:3.11-slim

# Creamos un usuario sin privilegios llamado 'appuser' por seguridad
RUN useradd -m -r appuser

WORKDIR /app

# Copiamos las dependencias instaladas desde la etapa de construcción
COPY --from=builder /install /usr/local

# Copiamos el código fuente y cambiamos el propietario al 'appuser'
COPY --chown=appuser:appuser . .

# Cambiamos al usuario sin privilegios
USER appuser

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]