# Usa una imagen oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto por defecto de FastAPI
EXPOSE 8080

# Comando para ejecutar la API con uvicorn
CMD ["uvicorn", "app.API:app", "--host", "0.0.0.0", "--port", "8080"]