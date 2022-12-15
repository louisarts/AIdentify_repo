# AIdentify

---

## Table of contents
* [General info](#general-info)
* [Getting the data](#getting-data)
* [Preparing the Data](#preparing-data)
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

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/static/img/page_imgs/fer_emotions.png" height="200px">

---

## Preparing the Data

1. Data Cleaning

The first step in preparing our data was to clean it. Cleaning data is an important part of any data preparation process – it usually involves correcting or removing incorrect, wrongly formatted, duplicated or incomplete data within a dataset.

In our case data cleaning involved dropping clearly mistaken images (i.e., those not showing faces), and dropping images where the labels did not seem appropriate (e.g., a face showing surprise but labelled as “sad”).

2. Resizing & Haar Cascade filtering

The next step in our data preparation process involved using the Haar cascade filter to detect and then essentially crop out the face from the images (i.e., getting rid of the background). Then we resized the images so that they had a uniform standard size (i.e., 200x200 pixels).

3. Grayscaling/Normalizing

Next, we made sure to change all images from color (RGB) to grayscale (one color channel). We also normalized the pixel intensity by dividing each pixel by 255 (each color has 255 parameter values based on color intensity).

4. Augmentation (mirrroing & rotating)

Since the dataset was slightly imbalanced in favor of happy, neutral, and surprised emotions, we wanted to boost the size of other emotion classes (e.g., fear, disgust, contempt). We did this by augmenting the already existing data by mirroring and rotating the underrepresented emotion class images. This allowed us to have a more balanced dataset for training the models.

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/static/img/page_imgs/data_prep.png" height="500px">



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
