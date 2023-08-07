import os
import cv2
import json
import time
import datetime
import threading
from util import colors
import pandas as pd
import Jetson.GPIO as GPIO
from modzy import EdgeClient
from modzy.edge import InputSource
from flask import Response, Flask, render_template
import Jetson.GPIO as GPIO

# Define constant variables to be used in workflow
CAPTURE_WIDTH=1920
CAPTURE_HEIGHT=1080
DISPLAY_WIDTH=960
DISPLAY_HEIGHT=540
FRAMERATE=30
FLIP_METHOD=2
# Workflow parameters
STREAM = False
SAVE_VID = True
MODEL_ID = "<model-identifier>"
MODEL_VERSION = "<model-version>"
NAMES = [] # e.g., ['broken_teeth', 'dent', 'scratch']
NUM_FRAMES = 90
lw = 2
tf = max(lw - 1, 1)
txt_color=(255, 255, 255)

# Set up the GPIO pins
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Define GPIO pins used to toggle each LED color
LED_PINS = {"red":29, "green":15, "white":7}
# Red: "Defect Detected", Green: "Model Running", White: "Core Status"
channels = [LED_PINS["red"], LED_PINS["green"], LED_PINS["white"]]
GPIO.setup(channels, GPIO.OUT)

# Define functions for light manipulation
def turn_on(color):
    if color not in LED_PINS:
        raise ValueError("LED color must be one of %r." % LED_PINS)
    else:
        GPIO.output(LED_PINS[color], GPIO.HIGH)

def turn_off(color):
    if color not in LED_PINS:
        raise ValueError("LED color must be one of %r." % LED_PINS)
    else:
        GPIO.output(LED_PINS[color], GPIO.LOW)

def pulse(color, times=1):
    if color not in LED_PINS:
        raise ValueError("LED color must be one of %r." % LED_PINS)
    else:
        for _ in range(times):
            GPIO.output(LED_PINS[color], GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(LED_PINS[color], GPIO.LOW)

# Image frame sent to the Flask object
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# Tables to show real-time prediction analysis
global aggregate_frame_df
aggregate_frame_df = pd.DataFrame(columns=["Timestamp", "Defect", "Confidence Score"])

# Create the Flask object for the application
app = Flask(__name__)

# Define function for connecting to gstreamer video pipeline
def gstreamer_pipeline(
    exposuretime_low = 8000000,
    exposuretime_high = 8000000,
    capture_width=CAPTURE_WIDTH,
    capture_height=CAPTURE_HEIGHT,
    display_width=DISPLAY_WIDTH,
    display_height=DISPLAY_HEIGHT,
    framerate=FRAMERATE,
    flip_method=FLIP_METHOD,
):
    return (
        "nvarguscamerasrc exposuretimerange=\"%d %d\" ! "
        "video/x-raw(memory:NVMM), "   
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True"
        % (
            exposuretime_low,
            exposuretime_high,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def capture_frames():
    global video_frame, thread_lock, aggregate_frame_df

    # Video capturing from OpenCV
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    start_time = time.time()
    client = EdgeClient("localhost", 55000)
    pulse("white",3)
    turn_on("white")
    client.connect()
    while True and video_capture.isOpened():
        try:
            return_key, frame = video_capture.read()
            if not return_key:
                print("no return key")
                break

            # Create a copy of the frame and store it in the global variable,
            # with thread safe access
            with thread_lock:
                '''
                Frame preparation for Modzy
                '''
                _, encoded_img = cv2.imencode('.jpg', frame)
                input_object = InputSource(
                    key="image",
                    data=encoded_img.tobytes()
                )       
                
                '''
                Modzy APIs
                '''
                turn_on('green')
                inference = client.inferences.run(MODEL_ID, MODEL_VERSION, [input_object])
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if len(inference.result.outputs.keys()):
                    results = json.loads(inference.result.outputs['results.json'].data)
                    frame_stats = [[timestamp, det['class'], det['score']] for det in results['data']['result']['detections']]
                    frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Defect", "Confidence Score"])
                    aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)
                    '''
                    Postprocessing
                    '''
                    preds = results['data']['result']['detections']

                    if len(preds):
                        pulse('red', 1)
                        for det in preds:
                            p1, p2 = (det['xmin'], det['ymin']), (det['xmax'], det['ymax'])
                            label = det['class']
                            color = colors(NAMES.index(label), True)
                            # plot bboxes first
                            cv2.rectangle(frame, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                            # then add text
                            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
                            outside = p1[1] - h >= 3
                            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                            cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)
                            cv2.putText(frame,
                                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                        0,
                                        lw / 3,
                                        txt_color,
                                        thickness=tf,
                                        lineType=cv2.LINE_AA)
                    video_frame = frame.copy()
                else:
                    continue       
                
            
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        except Exception as e:
            turn_off("green")
            turn_off("white")
            print("ERROR:\n{}".format(e.with_traceback()))
            break
    end_time = time.time()
    print(f'Video Stream closed after {end_time - start_time} seconds')
    video_capture.release()
    client.close()
    turn_off("green")
    turn_off("red")
    turn_off("white")
        
def encode_frame():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            global video_frame
            if video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')

def format_table(num_rows: int=20):
    global thread_lock, aggregate_frame_df
    with thread_lock:
        if aggregate_frame_df.shape[0] >=num_rows:
            aggregate_frame_df = aggregate_frame_df[:num_rows]
    return aggregate_frame_df.to_html(index=False)

@app.route('/')
def index():
    return render_template('app.html', tables=format_table(25), titles=None)

@app.route("/table_data")
def return_table():
    return format_table(25)

@app.route("/stream_frames")
def stream_frames():
    return Response(encode_frame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':

    # Create a thread and attach the method that captures the image frames, to it
    process_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread.start()
    
    # create thread for table creation
    # table_thread = threading.Thread(target=)

    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    app.run(host="0.0.0.0", port="8000")
