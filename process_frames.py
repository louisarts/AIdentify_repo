import cv2
from query_to_api import make_query
from process_emotions import process_emotions, initialize_data_frame
from process_emotions import  save_data_frame,corrector,apply_corrector


#Input: number of frames, image from the webcam, label from previous image,
#model to use for the prediction

def process_frames(i,frame,label,model):
    #Initialize the data frame
    file = "data1.csv"
    if i == 0:
        initialize_data_frame(file,model)
    faceCascade = cv2.CascadeClassifier('cascade.xml')
    # Capture frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        # Draw a rectangle around all the faces that appear on the frame

    for (x, y, w, h) in faces:
        # Query the emotion of the face with the query_to_api function
        results = make_query(frame[y:y + h, x:x + w], model)
        save_data_frame(process_emotions(results,i),file)
        label = results.json()["Emotion"]
        # Draw a rectangle over each of the faces that appear on the frame each
        #with it's emotion
        if model == "model1":
            color = (0,0,0)
        if model =='model2':
            color = (102,85,0)
        if model =='model3':
            color = (0,115,230)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        # Put a label on each of the face
        fontScale = (w/200.)
        rect_heigh = int(35 * fontScale)
        cv2.rectangle(frame, (x, y), (x+w, y-rect_heigh), color, -1)

        cv2.putText(frame, text=label, org=(x+10,y-10),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255),
            thickness=2, lineType=cv2.LINE_AA)
    i += 1
    return i, frame, label
