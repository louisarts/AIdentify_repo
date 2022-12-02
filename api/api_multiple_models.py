from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from tensorflow import keras
#For Model and Model2:
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', "Surprise"]
# For DenseNet169:
#CLASS_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
my_app = FastAPI()
print("Loading model...")
model1 = keras.models.load_model("../model2")
model2 = keras.models.load_model("../DenseNet169")
print("Model loaded")
preprocess_fun = keras.applications.densenet.preprocess_input

class Info(BaseModel):
    image : list[list[list[int]]]

# we do use the image to predict the emotion. Input: image, Output, str
@my_app.post("/predict_model1")
def predict(info : Info):
    #preprocess the image might need to be changed depending on the final model
    #we need first to convert the list into a np.array
    frame_prepros = preprocess_fun(np.array(info.image))
    #normalize the values of the image
    frame_rescaled=frame_prepros/255
    #add nex axis
    fr = frame_rescaled[np.newaxis]
    #predict the emotion
    predict = model1.predict(fr)
    #convert the value to a label
    label = CLASS_LABELS[np.argmax(predict)]
    return label

@my_app.post("/predict_model2")
def predict(info : Info):
    #preprocess the image might need to be changed depending on the final model
    #we need first to convert the list into a np.array
    frame_prepros = preprocess_fun(np.array(info.image))
    #normalize the values of the image
    frame_rescaled=frame_prepros/255
    #add nex axis
    fr = frame_rescaled[np.newaxis]
    #predict the emotion
    predict = model2.predict(fr)
    #convert the value to a label
    label = CLASS_LABELS[np.argmax(predict)]
    return label
