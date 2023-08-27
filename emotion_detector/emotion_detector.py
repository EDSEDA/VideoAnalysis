from datetime import datetime

from keras.models import load_model
import cv2
import numpy as np
import time
import json
import configparser
from threading import Thread, Lock

from api.rabbit import mq_send
from api.config import EMOTION_LABELS, paths

face_classifier = cv2.CascadeClassifier(paths.FACE_CLASSIFIER_PATH)  # детектор лица OpenCV
classifier = load_model(paths.PREDICTION_MODEL_PATH)  # обученная модель для классификации эмоций

config = configparser.ConfigParser()
config.read(paths.CONFIG_PATH)
video_driver_path = config['DEFAULT']['video_driver_path']
send_period_ms = int(config['DEFAULT']['send_period_ms'])
worker_id = int(config['DEFAULT']['worker_id'])

print(video_driver_path)
cap = cv2.VideoCapture(video_driver_path)

FRAME_RATE = 10
FACE_CLASSIFIER_MIN_NEIGHBORS=10
FACE_CLASSIFIER_MIN_SIZE=(40, 40)

mutex = Lock()

emotions_buffer = dict.fromkeys(EMOTION_LABELS, 0)
emotions_buffer["worker_id"] = worker_id

def try_detect_frame():
    prev = 0
    while True:

        time_elapsed = time.time() - prev
        _, frame = cap.read()

        if time_elapsed > 1. / FRAME_RATE:
            prev = time.time()

            # faces searching
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            faces = face_classifier.detectMultiScale(frame, minNeighbors=FACE_CLASSIFIER_MIN_NEIGHBORS, minSize=FACE_CLASSIFIER_MIN_SIZE)

            for (x, y, w, h) in faces:

                # face cutting
                cv2.rectangle(frame, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 255), 2)
                roi_box = frame[y:y + h, x:x + w]

                # some image processing and kostyls
                roi_box = cv2.resize(roi_box, (48, 48), interpolation=cv2.INTER_AREA)
                roi = np.empty((1, 48, 48, 3))
                roi[0] = roi_box
                roi = roi / 255

                #prediction making
                prediction = classifier.predict(roi)
                emotion_label = EMOTION_LABELS[np.argmax(prediction)]
                with mutex:
                    emotions_buffer[emotion_label] += 1

                # prediction drawing
                label_position = (x, y)
                cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def send_buffer():
    while True:
        time.sleep(send_period_ms)
        emotions_buffer["date"] = int(datetime.now().timestamp())
        with mutex:
            mq_send(json.dumps(emotions_buffer))
            for key in EMOTION_LABELS:
                emotions_buffer[key]= 0

sendThread = Thread(target=send_buffer)
sendThread.start()

try_detect_frame()
cap.release()
cv2.destroyAllWindows()
