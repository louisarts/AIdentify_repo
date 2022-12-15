<!--div align="center">
    <h1>AIdentify</h1>
</div-->

--- 

AIdentify is a data science project, built in less than 2 weeks, which, given a live video input, identifies faces and classifies the emotions being expressed . The project intends to demonstrate numerous facets of data science, from data preparation to model selection and training. There are numerous potential applications of our project, including measuring reactions to digital advertisements, improving eLearning, and getting feedback on digital entertainment experiences.

We have developed an app written in Python, served as a Flask application. 

---

<div align="center">
    <h2> Table of Contents </h2>
</div>


## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)




+-+-+-+-+-+-+-+-+-+-+-
Run pip install -e . to install
+-+-+-+-+-+-+-+-+-+-+-
To run the code:
On folder api run: uvicorn api_multiple_models:my_app
When the model is loaded:
1) If we want to show locally:
    python video_capture_with_prediction.py
2) If we want to show on the browser:
    python main.py
   Open browser on localhost:5000
