# AIdentify

---

## Table of contents
* [General info](#general-info)
* [Getting the data](#getting-data)
* [Technologies](#technologies)
* [Setup](#setup)

---

## General info
AIdentify is a data science project, built in less than 2 weeks, which, given a live video input, identifies faces and classifies the emotions being expressed . The project intends to demonstrate numerous facets of data science, from data preparation to model selection and training. There are numerous potential applications of our project, including measuring reactions to digital advertisements, improving eLearning, and getting feedback on digital entertainment experiences.

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/static/img/page_imgs/jim_faces.jpg" height="500px">

---

## Getting the data
Through our research process we were able to identify two high-quality datasets with labels which matched what we wanted our model to classify. These two datasets were FER+ and AffectNet-HQ . These two datasets are made openly available through the Kaggle platform, and we give credit to the authors who first introduced them on this page.

The FER+ (Face Expression Recognition Plus) dataset is an extension/correction of the original FER-2013 dataset, which contained approximately 30,000 images of faces that are grayscaled and have been uniformly sized to 48x48 pixels. The FER+ dataset essentially takes the original FER-2013 dataset and re-labels the images to classify them into one of eight emotions: neutral, happy, sad, surprised, fearful, angry, disgust and contempt. This relabeling was down by 10 crowd-sourced taggers, which makes the labeling process more robust as it doesn’t just depend on the evaluation of a single labeler (as was the case in the original 2013 version of the dataset).

The AffectNet-HQ dataset is an annotated database of facial expressions “in the wild” (which are essentially candid photos which are not staged just for the benefit of the dataset). The database was created by collecting facial images from the internet by querying three major search engines using a large number of emotion related keywords in six different languages. The version of the database available on Kaggle has been cleaned to relabel the emotional classes of certain images and utilizes an image enhancement technique to improve image quality. In total the dataset contains about 31,000 images of varying pixel sizes.

<img src=""

---

## Technologies
We have developed an app written in Python (version: 3.10.6), served as a Flask application. 
We have used the following libraries:

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
