import cv2
import numpy as np
import time
import json
import yaml
import sys
sys.path.append('..')

from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from keras.models import load_model
from datetime import datetime
from threading import Thread, Lock
from api.rabbit import mq_send
from api.config import EMOTION_LABELS, paths
from ultralytics import YOLO

from pathlib import Path

REAL_FPS=5
FACE_CLASSIFIER_MIN_NEIGHBORS=12
FACE_CLASSIFIER_MIN_SIZE=(56, 56)

modelYolo = YOLO('../models/yolov8n-face.pt')

def try_detect_frame(worker_id: int, video_driver_path: str, cap: any, client_number: int):

    while (True):
        is_frame, image_full = cap.read()

        if is_frame == False:
            print("no frame")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("YOLOv8 Tracking cropped", image_full)

        results = modelYolo.track(image_full, persist=True)
        # print(results)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking cropped", annotated_frame)

with open(paths.CONFIG_PATH, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


send_period_s = int(config['send_period_s'])
worker = config['workers'][0]
worker_id = int(worker['id'])
video_driver_path = worker['video_driver_path']
cap = cv2.VideoCapture(video_driver_path)

try_detect_frame(worker_id, video_driver_path, cap, 1)

print("end")
cv2.destroyAllWindows()
