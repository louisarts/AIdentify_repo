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

def make_query(image):
    dsize = (75, 75)
    #reescale the image
    reduced_image = cv.resize(image,dsize)
    url= "http://localhost:8000/predict"
    data = {
        "image" : reduced_image.tolist()
    }
    json_data = json.dumps(data)
    response = requests.post(url, data=json_data).text
    return response
