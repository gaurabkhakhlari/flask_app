from fastapi import FastAPI, HTTPException, UploadFile, Form, File
import base64
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

app = FastAPI()

model = load_model('cnnXh5')

ROOT_FOLDER = "predicted_images"

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

@app.post("/app")
async def upload_bitmap(image: str = Form(...), name: str = Form(...)):
    try:
        image_bytes = base64.b64decode(image)
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        image = image.convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        threshold = 127
        image_array = (image_array > threshold) * 255
        image_array = 255 - image_array
        image_array = image_array.reshape((1, 28, 28, 1))
        image_array = image_array / 255.0
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class])
        confidence *= 100
        class_name = str(predicted_class + 1)
        
        create_folder_if_not_exists(ROOT_FOLDER)
        
        class_folder_path = os.path.join(ROOT_FOLDER, class_name)
        create_folder_if_not_exists(class_folder_path)
        
        image_path = os.path.join(class_folder_path, name + "_processed.jpg")
        image.save(image_path)
        
        print("Predicted Class:", class_name)
        print("Confidence:", confidence)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {"message": int(predicted_class + 1), "confidence": confidence}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
