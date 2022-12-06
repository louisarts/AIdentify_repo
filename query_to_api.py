import requests
import numpy as np
import json
import cv2 as cv

#Function to query to our REST API sending as a post query the image reescaled
#As a response it gets the label of the emotion.
#Input: image : np.array
#Output: label: str

#With Model DenseNet169_125x125 use dsize = (125,125)
#With Model DenseNet169_75x75 use dsize = (75,75)
#With Model and Model2 use dsize =(48,48)

def make_query(image, model):
    if model == "model1":
        dsize = (48, 48)
        class_labels  = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', "Surprise"]
    elif model == "model2":
        dsize = (75,75)
        class_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    elif model == "model3":
        dsize = (48,48)
        class_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    #reescale the image
    reduced_image = cv.resize(image,dsize)
    url= "http://localhost:8000/predict_"+model
    data = {
        "image" : reduced_image.tolist(),
        "labels": class_labels
    }
    json_data = json.dumps(data)
    response = requests.post(url, data=json_data)
    return response
