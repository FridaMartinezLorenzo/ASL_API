from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from app.lsm_detector import detectar_letra_en_imagen



app = FastAPI(
    title="API de Detección de Letras en Lengua de Señas Americana (ASL)",
    description="""
    Esta API permite detectar letras del alfabeto en lenguaje de señas americano (ASL) a partir de una imagen de la mano.
    
    **Sube una imagen** con la mano mostrando una letra, y la API intentará reconocerla.
    """,
    version="1.0.0"
)

# Opcional: permitir llamadas desde otras apps/webs (ej. Flutter frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detectar", summary="Detectar letra en imagen", tags=["Detección"])
async def detectar(file: UploadFile = File(...)):
    """
    Sube una imagen (.jpg, .png, etc.) con una mano haciendo una seña de letra en ASL.
    
    Retorna la letra detectada, o `null` si no se detecta ninguna.

    - **file**: Imagen de la mano (formato JPEG/PNG).
    """
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        letra = detectar_letra_en_imagen(img)

        return JSONResponse(content={"letra_detectada": letra}, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"error": f"No se pudo procesar la imagen: {str(e)}"},
            status_code=500
        )
