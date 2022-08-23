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
    try:
        contents = file.file.read()
        with open("images/" + file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    image = np.asarray(load_img("images/" + file.filename, target_size=(225, 400))).astype('float32') / 255
    image = expand_dims(image, axis=0) 

    score, acc = model.evaluate(image,np.array([0]),verbose=0, batch_size=1)
    if(acc >= 0.99):
        shutil.move("./images/" + file.filename, "images/Car/" + file.filename)
        return {"car": True, "score": score, "accuracy" : acc}
    shutil.move("./images/" + file.filename, "images/NoCar/" + file.filename)    
    return {"car": False, "score": score, "accuracy" : acc}