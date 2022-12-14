from flask import Flask, render_template, request,Response,url_for
import cv2,imutils,time
import pyshine as ps
from process_frames import process_frames

app = Flask(__name__,template_folder='templates')

def pyshine_process(model):
    global cap
    cap = cv2.VideoCapture(0)
    fps=0
    st=0
    frames_to_count=20
    cnt=0
    label = "Happiness"
    i = 0
    labels = []
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            if cnt == frames_to_count:
                fps = round(frames_to_count/(time.time()-st))
                st = time.time()
                cnt=0
            i, img, label1 = process_frames(i,img, label, model)
            if label1 != label:
                label = label1
            if label1 not in labels:
                cv2.imwrite(f'output/{label}.png',img)
                labels.append(label1)
            frame = cv2.imencode('.JPEG', img,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tobytes()
            #time.sleep(0.016)

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break



@app.route('/')
def index():
   return render_template('index.html')

@app.route('/features1')
def features1():
    return render_template("/overview.html")

@app.route('/features2')
def features2():
    return render_template("/data1.html")

@app.route('/features3')
def features3():
    return render_template("/data2.html")

@app.route('/visualize')
def visualize():
    return render_template("/visualize.html")

@app.route('/cnn')
def cnn():
    return render_template("/cnn.html")

@app.route('/modelselection')
def modelselection():
    return render_template("/modelselection.html")

@app.route('/team')
def team():
    return render_template("/team.html")

@app.route('/mod1')
def mod1():
    return render_template("/model1.html")

@app.route('/model1')
def model1():
    model ="model1"
    return Response(pyshine_process(model),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mod2')
def mod2():
    return render_template("/model2.html")

@app.route('/model2')
def model2():
    model = "model2"
    return Response(pyshine_process(model),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mod3')
def mod3():
    return render_template("/model3.html")

@app.route('/model3')
def model3():
    model = "model3"
    return Response(pyshine_process(model),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template("/about.html")

@app.route('/furtherinfo')
def furtherinfo():
    return render_template("/furtherinfo.html")

@app.route('/kill_camera')
def kill_camera():
    global cap
    cap.release()
    return render_template("/modelselection.html")

@app.route('/kill_camera2')
def kill_camera2():
    global cap
    cap.release()
    return render_template("/show_result.html")

if __name__ == "__main__":
    app.run(debug=True,threaded=True)
