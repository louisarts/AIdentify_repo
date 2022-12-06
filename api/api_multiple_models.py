from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from tensorflow import keras
from skimage import color

my_app = FastAPI()
print("Loading model 1...")
model1 = keras.models.load_model("../model_0.7107.h5")
print("Loading model 2...")
model2 = keras.models.load_model("../DenseNet169_75x75")
print("Loading model 3...")
model3 = keras.models.load_model("../DenseNet169_48x48")
print("Models loaded")
preprocess_fun = keras.applications.densenet.preprocess_input

class Info(BaseModel):
    image : list[list[list[int]]]
    labels: list[str]

# we do use the image to predict the emotion. Input: image, Output, str
@my_app.post("/predict_model1")
def predict(info : Info):
    labels = info.labels
    #preprocess the image might need to be changed depending on the final model
    #we need first to convert the list into a np.array
    gray = color.rgb2gray(np.array(info.image))

    frame_prepros = preprocess_fun(gray)
    #normalize the values of the image
    frame_rescaled=frame_prepros/255
    #add nex axis
    fr = frame_rescaled[np.newaxis]
    #predict the emotion
    predict = model1.predict(fr).tolist()
    v_m = max(predict[0])
    maxp = predict[0].index(v_m)
    #Convert the predict list into a dictionary
    response = {'Anger': predict[0][0],
                'Contempt':predict[0][1],
             'Disgust': predict[0][2],
             'Fear': predict[0][3],
             'Happiness': predict[0][4],
             'Neutral': predict[0][5],
             'Sadness': predict[0][6],
             'Surprise': predict[0][7],
             "Emotion": labels[maxp]}

    return response

@my_app.post("/predict_model2")
def predict(info : Info):
    labels = info.labels
    #preprocess the image might need to be changed depending on the final model
    #we need first to convert the list into a np.array
    frame_prepros = preprocess_fun(np.array(info.image))
    #normalize the values of the image
    frame_rescaled=frame_prepros/255
    #add nex axis
    fr = frame_rescaled[np.newaxis]
    #predict the emotion
    predict = model2.predict(fr).tolist()
    v_m = max(predict[0])
    maxp = predict[0].index(v_m)
    #Convert the predict list into a dictionary
    print("len vector",len(predict))
    response = {'Anger': predict[0][0],
                'Contempt':predict[0][1],
             'Disgust': predict[0][2],
             'Fear': predict[0][3],
             'Happiness': predict[0][4],
             'Neutral': predict[0][5],
             'Sadness': predict[0][6],
             'Surprise': predict[0][7],
             "Emotion": labels[maxp]}
    return response

@my_app.post("/predict_model3")
def predict(info : Info):
    labels = info.labels
    #preprocess the image might need to be changed depending on the final model
    #we need first to convert the list into a np.array
    frame_prepros = preprocess_fun(np.array(info.image))
    #normalize the values of the image
    frame_rescaled=frame_prepros/255
    #add nex axis
    fr = frame_rescaled[np.newaxis]
    #predict the emotion
    predict = model2.predict(fr).tolist()
    v_m = max(predict[0])
    maxp = predict[0].index(v_m)
    print(len(predict))
    #Convert the predict list into a dictionary
    response = {'Anger': predict[0][0],
                'Contempt':predict[0][1],
             'Disgust': predict[0][2],
             'Fear': predict[0][3],
             'Happiness': predict[0][4],
             'Neutral': predict[0][5],
             'Sadness': predict[0][6],
             'Surprise': predict[0][7],
             "Emotion": labels[maxp]}

    return response
