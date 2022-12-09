import pandas as pd

def process_emotions(result, index):
    data = pd.DataFrame(result.json(), index=[index])
    return data


def initialize_data_frame(file,model):
    class_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise',"Emotion"]

    df = pd.DataFrame(columns = class_labels)
    df.to_csv(file, index=False)
    return "successfully created"

def save_data_frame(data, file):
    data.to_csv(file, mode='a', index=False, header=False)
    return "successfully appended"


# Corrector from neutral state.
def corrector(file):
    data = pd.read_csv(file)
    corrector = data.mean()
    return corrector

def apply_corrector(data, corrector, model):
    data.Disgust = data.Disgust - corrector.Disgust
    if model == "model3" or model == "model2":
        data.Contempt = data.Contempt -corrector.Contempt
    data.Fear = data.Fear - corrector.Fear
    data.Happiness = data.Happiness - corrector.Happiness
    data.Sadness = data.Sadness - corrector.Sadness
    data.Surprise = data.Surprise - corrector.Surprise
    return data
