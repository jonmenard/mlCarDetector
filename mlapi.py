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
from numpy import expand_dims,asarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import shutil
import io
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
    try:
    
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).resize((400,225), Image.Resampling.NEAREST)
        image = np.asarray(image).astype('float32') / 255
        image = expand_dims(image, axis=0) 
        score, acc = model.evaluate(image,np.array([0]),verbose=0, batch_size=1)
        print(acc)
        if(acc >= 0.99):
            car = True
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        
    return {"car": car, "score": score, "accuracy" : acc}