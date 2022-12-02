import cv2
from query_to_api import make_query
from most_common import most_common

faceCascade = cv2.CascadeClassifier('cascade.xml')

video_capture = cv2.VideoCapture(0)
i = 0
result = "Happiness"
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around all the faces that appear on the frame

    for (x, y, w, h) in faces:
        # Query the emotion of the face with the query_to_api function
        #if (i % 15 == 0):
        result = make_query(frame[y:y + h, x:x + w]).replace('"',"")

        # Draw a rectangle over each of the faces that appear on the frame each
        #with it's emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)
        # Put a label on each of the face
        fontScale = (w/200.)
        rect_heigh = int(35 * fontScale)
        cv2.rectangle(frame, (x, y), (x+w, y-rect_heigh), (0, 0, 0), -1)

        cv2.putText(frame, text=result, org=(x+10,y-10),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255),
            thickness=2, lineType=cv2.LINE_AA)
    i += 1
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
