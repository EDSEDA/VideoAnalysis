import cv2
import numpy as np
import time
import json
import yaml
import sys
sys.path.append('..')

# from keras.models import load_model
# from datetime import datetime
# from threading import Thread, Lock
# from api.rabbit import mq_send
from api.config import EMOTION_LABELS, paths

import predictor

with open(paths.CONFIG_PATH, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


send_period_s = int(config['send_period_s'])
# for worker in config['workers']:
#     worker_id = int(worker['id'])
#     video_driver_path = worker['video_driver_path']
#     Thread(target=try_detect_frame, kwargs={'worker_id': worker_id, 'video_driver_path': video_driver_path}).run()

worker = config['workers'][0]
worker_id = int(worker['id'])
gst_stream = worker['gst_stream']
cap = cv2.VideoCapture(gst_stream, cv2.CAP_GSTREAMER)

predictor.try_detect_frame(worker_id, cap)
print("cv detector was closed")
