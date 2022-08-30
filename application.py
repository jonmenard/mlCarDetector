from fastapi import FastAPI, Request, File, UploadFile
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
# from starlette.responses import StreamingResponse
model = keras.models.load_model(".mdl_wts.hdf5")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploading")
async def upload(file: UploadFile = File(...)):
    score = 0
    acc = 0
    car = False
    # try:
    directory = "C:/Users/gasma/Documents/GitHub/SYSC5108/Dataset/Dataset10/trainingData/cars" 
    filename = file.filename
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents)).resize((400,225), Image.Resampling.LANCZOS)
    if not filename.endswith(".jpg"):
        image = image.convert('RGB')
        file.filename += ".jpg" 

    
    # image = load_img(directory +"/" + filename)
    image1 = np.asarray(image).astype('float32') / 255
    image1 = expand_dims(image1, axis=0)
    if(len(image1.shape) < 4):
        image = image.convert('RGB')
        image1 = np.asarray(image).astype('float32') / 255
        image1 = expand_dims(image1, axis=0)


    print(image1.shape)
    score = model(image1, training = False).numpy().flatten()
    print(score[0])
    
    # except Exception:
    file.file.close()

    num1 = random.randint(0, 9)
    num2 = random.randint(0, 9)
    image.save("images/saved/" + str(num1) + str(num2) + "_" + file.filename)

    if(score[0] < 0.50):
        car = True
    # return {"message": "There was an error uploading the file"}
    # finally:
    
    return {"car": car, "score": float(score[0]), "accuracy" : acc}

if __name__=="__main__":
    uvicorn.run("application:app", reload=True, debug=True, workers=2)