# AIdentify

---

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

---

## General info
AIdentify is a data science project, built in less than 2 weeks, which, given a live video input, identifies faces and classifies the emotions being expressed . The project intends to demonstrate numerous facets of data science, from data preparation to model selection and training. There are numerous potential applications of our project, including measuring reactions to digital advertisements, improving eLearning, and getting feedback on digital entertainment experiences.

---

## Technologies
We have developed an app written in Python, served as a Flask application. 

* numpy 1.23.5
* opencv-python 4.6.0.66
* requests 2.28.1
* tensorflow 2.11.0
* pydantic 1.10.2
* fastapi 0.88.0
* uvicorn 0.20.0
* jinja2 3.1.2
* flask 2.2.2
* imutils 0.5.4
* pyshine 0.0.9
* statistics 1.0.3.5
* scikit-image 0.19.3
* gunicorn 20.1.0


---

## Setup

To install this project, first:

1) Run pip install -e . to install

To run the code:

2) On folder api run: uvicorn api_multiple_models:my_app

When the model is loaded:

3.a) If we want to show locally:
    python video_capture_with_prediction.py
3.b) If we want to show on the browser:
    python main.py
   Open browser on localhost:5000
