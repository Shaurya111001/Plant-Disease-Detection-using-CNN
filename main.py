from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps
import numpy 

app = FastAPI()
model_dir = "InceptionV3Net.h5"
model = load_model(model_dir)

class_predictions = array(['Strawberry___healthy', 'Tomato___Early_blight', 'Pepper,_bell___Bacterial_spot', 'Tomato___Late_blight', 'Apple___Cedar_apple_rust', 'Apple___Apple_scab', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Potato___healthy', 'Peach___Bacterial_spot', 'Squash___Powdery_mildew', 'Tomato___Target_Spot', 'Apple___healthy', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Grape___Black_rot', 'Apple___Black_rot', 'Potato___Late_blight', 'Cherry_(including_sour)___Powdery_mildew', 'Grape___healthy', 'Blueberry___healthy', 'Raspberry___healthy', 'Strawberry___Leaf_scorch', 'Soybean___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Pepper,_bell___healthy', 'Tomato___Bacterial_spot', 'Cherry_(including_sour)___healthy', 'Tomato___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Peach___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Early_blight', 'Orange___Haunglongbing_(Citrus_greening)', 'Corn_(maize)___healthy', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot'])

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}


@app.post("/net/image/prediction/")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    
    img_path = get_file(
        origin = image_link
    )
    img = load_img(
        img_path, 
        target_size = (224, 224)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    pred = model.predict(img_array)
    score =softmax(pred[0])
    print(score)

    class_prediction = class_predictions[argmax(score)]

    model_score = round(max(score) * 100, 2)
    model_score = dumps(model_score.tolist())

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score
    }
    
# if __name__ == "__main__":
# 	port = int(os.environ.get('PORT'))
# 	run(app, host="0.0.0.0", port=port)