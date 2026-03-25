from fastapi import FastAPI, UploadFile
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Keras 3 way to load a SavedModel for inference
model = keras.layers.TFSMLayer('fashion_mnist_savedmodel', call_endpoint='serving_default')

CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.get("/")
def home():
    return {"message": "Fashion MNIST Classifier is running!"}

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert('L').resize((28, 28))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 28, 28, 1).astype('float32')

    # Get raw output
    preds = model(arr)
    output = list(preds.values())[0].numpy()

    # Convert logits to probabilities
    probabilities = np.exp(output) / np.sum(np.exp(output))

    predicted_class = CLASS_NAMES[np.argmax(probabilities)]
    confidence = float(np.max(probabilities))

    return {
        "label": predicted_class,
        "confidence": round(confidence, 4)
    }