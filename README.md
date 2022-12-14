# AIdentify

---

## Table of contents
* [General info](#general-info)
* [Getting the data](#getting-the-data)
* [Preparing the Data](#preparing-the-data)
* [Training the Model](#training-the-model)
* [Technologies](#technologies)
* [Architecture](#architecture)
* [Setup](#setup)
* [The Team](#the-team)

---

## General info
AIdentify is a data science project, built in less than 2 weeks, which, given a live video input, identifies faces and classifies the emotions being expressed . The project intends to demonstrate numerous facets of data science, from data preparation to model selection and training. There are numerous potential applications of our project, including measuring reactions to digital advertisements, improving eLearning, and getting feedback on digital entertainment experiences.

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/page_imgs/jim_faces.jpg" height="500px">

---

## Getting the data
Through our research process we were able to identify two high-quality datasets with labels which matched what we wanted our model to classify. These two datasets were FER+ and AffectNet-HQ . These two datasets are made openly available through the Kaggle platform, and we give credit to the authors who first introduced them on this page.

The FER+ (Face Expression Recognition Plus) dataset is an extension/correction of the original FER-2013 dataset, which contained approximately 30,000 images of faces that are grayscaled and have been uniformly sized to 48x48 pixels. The FER+ dataset essentially takes the original FER-2013 dataset and re-labels the images to classify them into one of eight emotions: neutral, happy, sad, surprised, fearful, angry, disgust and contempt. This relabeling was down by 10 crowd-sourced taggers, which makes the labeling process more robust as it doesn’t just depend on the evaluation of a single labeler (as was the case in the original 2013 version of the dataset).

The AffectNet-HQ dataset is an annotated database of facial expressions “in the wild” (which are essentially candid photos which are not staged just for the benefit of the dataset). The database was created by collecting facial images from the internet by querying three major search engines using a large number of emotion related keywords in six different languages. The version of the database available on Kaggle has been cleaned to relabel the emotional classes of certain images and utilizes an image enhancement technique to improve image quality. In total the dataset contains about 31,000 images of varying pixel sizes.

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/page_imgs/fer_emotions.png" height="200px">

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

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/page_imgs/data_prep.png" height="500px">

## Training the Models

We have trained two different models.

### Custom Convolutional Neural Network
The first model consisted on an entire Convolutional Neural Network (CNN) architecture designed by AIdentify developers. With this model we were able to predict the emotions with a 71% Accuracy.

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/page_imgs/cnn.png" height="500px">

### Model constructed using a pre-trained network
On a second approach, we have used a pre-trained a Convolutional Neural Network (Densenet169), where we have added a few extra layers.

We have trained the model in two sequences.
<ul><li>On the first step, we have frozen the pre-trained network layers and just trained the layers that we have added. </li>
    <li>On a second step, after the model has converged, we have finetuned the fitting by unfrozing a few layers from the pre-trained CNN, using very small steps.</li></ul>

Following this approach we have achieved a model that is able to predict the emotions with a 81% of accuracy.

<img src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/page_imgs/transfer_learning.png" height="200px">

## Technologies
We have developed our app in Python (version: 3.10.6), served as a Flask application.
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

## Architecture

Our app is deployed as a web page served using Flask. We have used Bootstrap for the layout of the webpage.

<div>
  <div>
    <img width="49%" src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/readme_imgs/index.png" style="margin-top: 8px; vertical-align: middle;">
    <img width="49%" src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/readme_imgs/project.png" style="margin-top: 8px; vertical-align: middle;">
    <img width="49%" src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/readme_imgs/model_select.png" style="margin-top: 8px; vertical-align: middle;">
    <img width="49%" src="https://raw.githubusercontent.com/louisarts/AIdentify_repo/main/aidentify/static/img/readme_imgs/key_learnings.png" style="margin-top: 8px; vertical-align: middle;">
  </div>
</div>

The program uses opencv to get the input from th computer's webcam. On each of the frames opencv is processing, we use a HaarCascade classifier to detect all the faces. Once the faces are detected, the pixels are send via post query to the FastAPI service for the chosen model. The API service predicts uses the model that is serving to predict what is the emotion that the face is showing and returns the prediction of the model as a list of probabilities. The emotion with a higher probability is then outputed on each of the frames as a label. Flask is used to outut the processed frames with the detected faces with the emotion labels as a video stream.

https://user-images.githubusercontent.com/45735538/208239343-1951fdba-d54b-4233-a447-8bae20566e62.mov




---

## Setup

To install this project, first:

1) Run pip install -e . to install

To run the code:

2) On folder api run: uvicorn api_multiple_models:my_app

When the model is loaded, on folder aidentify:

3.a) If we want to show locally:
<br>
    <ul>python video_capture_with_prediction.py</ul>
<br>
3.b) If we want to show on the browser:
<br>
<ul>    python main.py</ul>
   Open browser on localhost:5000

---

## The Team

### Alba Gutierrez Pedemonte
Alba has a background in Physics, worked as a consultant and is an entrepreneur</br>
<a href="https://www.linkedin.com/in/albagutierrezpedemonte/">LinkedIn</a><br>
<a href="mailto: albaguti@gmail.com">E-mail</a><br>
<a href ="https://github.com/albaguti">GitHub Profile</a>

### Louis Arts
Louis recently graduated with a Bachelor's in science (specialization in physics and mathematics) from Maastricht University. He plans to start a Master's in Astrophysics soon.<br>
<a href="https://www.linkedin.com/in/louis-arts-a768221a6/">LinkedIn</a></br>
<a href="mailto: artslouis@gmail.com">E-mail</a><br>
<a href ="https://github.com/louisarts">GitHub Profile</a>

### Patrick Hofbauer
Patrick has a Bachelor's in economics from UCLA and a Master's in Finance from the University of Innsbruck. He plans to apply his newfound data science skills in the financial industry.<br>
<a href="https://www.linkedin.com/in/patrick-hofbauer/">LinkedIn</a><br>
<a href="mailto: patrick.hofbauer15@gmail.com">E-mail</a><br>
<a href ="https://github.com/patrickhofbauer">GitHub Profile</a>
