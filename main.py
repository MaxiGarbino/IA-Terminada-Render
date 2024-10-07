from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Carga el modelo
model = load_model("models/clasificador_10_clases.h5")

# Definición de las clases
clases = [
    'piel-normal', 'lunar', 'melanoma', 'acne', 
    'carcinoma-de-celulas-escamosas', 'varicela', 'piel-quemada', 
    'queratosis-actinica', 'carcinoma-de-celulas-basales', 'queratosis-seborreica'
]

# Función para preprocesar la imagen
def preprocesar_imagen(imagen: Image.Image):
    try:
        img = imagen.resize((150, 150))  # Aseguramos que la imagen sea de 150x150
        x = np.array(img)
        x = np.expand_dims(x, axis=0)   # Añadir una dimensión extra para el batch
        x = x / 255.0                   # Normalización
        return x
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")

# Ruta para hacer predicciones
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contenido = await file.read()  # Leer el contenido del archivo
        imagen = Image.open(io.BytesIO(contenido))  # Abrir la imagen
    except Exception:
        raise HTTPException(status_code=400, detail="Error al leer el archivo de imagen.")

    try:
        x = preprocesar_imagen(imagen)  # Preprocesar la imagen
        predicciones = model.predict(x)  # Realizar la predicción

        predicted_class_idx = np.argmax(predicciones[0])  # Obtener el índice de la clase predicha
        probabilidad = predicciones[0][predicted_class_idx]  # Obtener la probabilidad
        clase_predicha = clases[predicted_class_idx]  # Obtener el nombre de la clase predicha

        return {
            "prediccion": clase_predicha,
            "probabilidad": f"{probabilidad * 100:.2f} %"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {e}")
