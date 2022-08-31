from fastapi import FastAPI, Request, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from typing import List, Union
# from pydantic import BaseModel
# import pickle
# import pandas as pd
# import os
import numpy as np
import uuid
from numpy import expand_dims,asarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import shutil
import io
import random
import os
import uvicorn
import os, json, boto3
from urllib.parse import urlparse

# from starlette.responses import StreamingResponse
model = keras.models.load_model(".mdl_wts.hdf5")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")\


AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

session = boto3.Session(
aws_access_key_id=AWS_ACCESS_KEY_ID,
aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)




@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/setCar/{status}/{filename}")
async def feedbackCar(status: str, filename : str):

    s3 = session.resource('s3')

    copy_source = {
    'Bucket': S3_BUCKET_NAME,
    'Key': ("temp/" + filename)
    }
    try:
        bucket = s3.Bucket(S3_BUCKET_NAME)
        bucket.copy(copy_source, status + "/cars/" + filename)
        print(filename, " is set to be a car")
        s3.Object(S3_BUCKET_NAME, "temp/" + filename).delete()
    except Exception as e:
        print("there was an error")
    # s3.Object(S3_BUCKET_NAME, "cars/" + filename).copy_from(CopySource=("temp/" + filename))
    return {"success": True}





@app.get("/setNotCar/{status}/{filename}")
async def feedbackNotCar(status: str, filename : str):
    s3 = session.resource('s3')

    copy_source = {
    'Bucket': S3_BUCKET_NAME,
    'Key': ("temp/" + filename)
    }

    bucket = s3.Bucket(S3_BUCKET_NAME)
    try:
        bucket.copy(copy_source, status + "/notCars/" + filename)
        print(filename, " is set to be a car")
        s3.Object(S3_BUCKET_NAME, "temp/" + filename).delete()
    except Exception as e:
         print("there was an error")
    return {"success": True}


@app.post("/uploading")
async def upload(file: UploadFile = File(...)):

    s3_client = session.client('s3')

    num1 = random.randint(0, 9)
    num2 = random.randint(0, 9)

    score = 0
    acc = 0
    car = False
    newName =  str(num1) + str(num2) + "_" + str(uuid.uuid4()) + ".jpg"
    object_name = os.path.basename(newName)
    
    try:
    # directory = "C:/Users/gasma/Documents/GitHub/SYSC5108/Dataset/Dataset10/trainingData/cars" 
        filename = file.filename
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).resize((400,225), Image.Resampling.LANCZOS)
        
        if not filename.endswith(".jpg"): # image is something like png
            image = image.convert('RGB')
            file.filename += ".jpg" 

        image1 = np.asarray(image).astype('float32') / 255
        image1 = expand_dims(image1, axis=0)
        
        if(len(image1.shape) < 4): # if image is grey scale
            image = image.convert('RGB')
            image1 = np.asarray(image).astype('float32') / 255
            image1 = expand_dims(image1, axis=0)

        score = model(image1, training = False).numpy().flatten()

        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format="JPEG")
        in_mem_file.seek(0)
        # image.save(newName)

        response = s3_client.upload_fileobj(in_mem_file , S3_BUCKET_NAME, "temp/" + object_name)
        print(response)
    finally: 
        file.file.close()

    
    

    if(score[0] < 0.50):
        car = True
    # return {"message": "There was an error uploading the file"}
    # finally:

    confidence = (score[0] - 0.5) / 0.5
    if(car):
        print("is car")
        confidence = (0.5 - score[0]) / 0.5
    confidence = round(confidence, 2) * 100
    print(confidence)
    return {"car": car, "score": float(score[0]), "confidence" : confidence, "filename" : newName}

if __name__=="__main__":
    uvicorn.run("application:app", reload=True, debug=True, workers=2)