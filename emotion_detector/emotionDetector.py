from keras.models import load_model
import cv2
import numpy as np
import time

import configparser

from api.rabbit import mq_send
from api.config import EMOTION_LABELS, paths

face_classifier = cv2.CascadeClassifier(paths.FACE_CLASSIFIER_PATH)  # детектор лица OpenCV
classifier = load_model(paths.PREDICTION_MODEL_PATH)  # обученная модель для классификации эмоций

config = configparser.ConfigParser()
config.read(paths.CONFIG_PATH)
video_driver_path = config['DEFAULT']['video_driver_path']
print(video_driver_path)
cap = cv2.VideoCapture(video_driver_path)

FRAME_RATE = 10
FACE_CLASSIFIER_MIN_NEIGHBORS=10
FACE_CLASSIFIER_MIN_SIZE=(40, 40)

def try_detect_frame():
    prev = 0
    while True:

        time_elapsed = time.time() - prev
        _, frame = cap.read()

        if time_elapsed > 1. / FRAME_RATE:
            prev = time.time()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            faces = face_classifier.detectMultiScale(frame, minNeighbors=FACE_CLASSIFIER_MIN_NEIGHBORS, minSize=FACE_CLASSIFIER_MIN_SIZE)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 255), 2)

                roi_box = frame[y:y + h, x:x + w]
                roi_box = cv2.resize(roi_box, (48, 48), interpolation=cv2.INTER_AREA)
                roi = np.empty((1, 48, 48, 3))
                roi[0] = roi_box
                roi = roi / 255

                prediction = classifier.predict(roi)
                emotion_label = EMOTION_LABELS[np.argmax(prediction)]
                label_position = (x, y)
                cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                mq_send(emotion_label)

            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


try_detect_frame()
cap.release()
cv2.destroyAllWindows()
