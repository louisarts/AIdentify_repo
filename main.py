from flask import Flask, render_template, request,Response
import cv2,imutils,time
import pyshine as ps
from process_frames import process_frames

app = Flask(__name__)
@app.route('/')
def index():
   return render_template('index.html')

def pyshine_process(params):
    cap = cv2.VideoCapture(0)
    fps=0
    st=0
    frames_to_count=20
    cnt=0
    label = "Happiness"
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            if cnt == frames_to_count:
                fps = round(frames_to_count/(time.time()-st))
                st = time.time()
                cnt=0
            img, label1 = process_frames(cnt,img, label)
            if label1 != label:
                label = label1
            frame = cv2.imencode('.JPEG', img,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tobytes()
            #time.sleep(0.016)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break


@app.route('/res',methods = ['POST','GET'])
def res():
	global result
	if request.method == 'POST':
		result = request.form.to_dict()
		return render_template("results.html",result = result)

@app.route('/results')
def video_feed():
	global result
	params= result
	return Response(pyshine_process(params),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,threaded=True)
