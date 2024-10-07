from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = load_model("models/clasificador_10_clases.h5")

clases = ['piel-normal', 'lunar', 'melanoma', 'acne', 'carcinoma-de-celulas-escamosas', 'varicela', 
          'piel-quemada', 'queratosis-actinica', 'carcinoma-de-celulas-basales', 'queratosis-seborreica']

def preprocesar_imagen(imagen: Image.Image):
    img = imagen.resize((150, 150))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contenido = await file.read
    imagen = Image.open(io.BytesIO(contenido))

    x = preprocesar_imagen(imagen)

    predicciones = model.predict(x)

    predicted_class_idx = np.argmax(predicciones[0])
    probabilidad = predicciones[0][predicted_class_idx]
    clase_predicha = clases[predicted_class_idx]

    return {
        "prediccion": clase_predicha,
        "probabilidad": f"{probabilidad * 100:.2f} %"
    }
