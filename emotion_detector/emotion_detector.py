
import cv2
import numpy as np
import time
import json
import yaml

from keras.models import load_model
from datetime import datetime
from threading import Thread, Lock
from api.rabbit import mq_send
from api.config import EMOTION_LABELS, paths

# Standard python libraries
import os
import time
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task

import joblib


FRAME_RATE = 10
FACE_CLASSIFIER_MIN_NEIGHBORS=10
FACE_CLASSIFIER_MIN_SIZE=(40, 40)

mutex = Lock()

def try_detect_frame(worker_id: int, video_driver_path: str):
    cap = cv2.VideoCapture(video_driver_path)
    workers[worker_id] = dict.fromkeys(EMOTION_LABELS, 0) # для каждого айдишника в словаре задаем словарь эмоций
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
                    workers[worker_id][emotion_label] += 1

                # prediction drawing
                label_position = (x, y)
                cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def send_buffer():
    while True:
        time.sleep(send_period_s)
        workers["date"] = int(datetime.now().timestamp())
        with mutex:
            mq_send(json.dumps(workers))
            # for value in workers.values():
            #     for key in EMOTION_LABELS:
                    # value[key] = 0

face_classifier = cv2.CascadeClassifier(paths.FACE_CLASSIFIER_PATH)  # детектор лица OpenCV
classifier = load_model(paths.PREDICTION_MODEL_PATH)  # обученная модель для классификации эмоций

with open(paths.CONFIG_PATH, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

workers = dict()

send_period_s = int(config['send_period_s'])
for worker in config['workers']:
    worker_id = int(worker['id'])
    video_driver_path = worker['video_driver_path']
    Thread(target=try_detect_frame, kwargs={'worker_id': worker_id, 'video_driver_path': video_driver_path}).start()

send_buffer()
cv2.destroyAllWindows()
