from keras.models import load_model
# from tensorflow.keras.applications.vgg19 import preprocess_input
import cv2
import numpy as np
import socket
import time

face_classifier = cv2.CascadeClassifier(r'../cfg/faceDetector.xml')  # детектор лица OpenCV
classifier = load_model("../cfg/model.h5")  # обученная модель для классификации эмоций

emotion_labels = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprized']

server = '127.0.0.1', 8181
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 8282))

cap = cv2.VideoCapture("/dev/video0")

frame_rate = 10
prev = 0
while True:

    time_elapsed = time.time() - prev
    _, frame = cap.read()

    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        labels = []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        faces = face_classifier.detectMultiScale(frame, minNeighbors=10, minSize=(40, 40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 255), 2)
            roi_box = frame[y:y + h, x:x + w]
            roi_box = cv2.resize(roi_box, (48, 48), interpolation=cv2.INTER_AREA)

            roi = np.empty((1, 48, 48, 3))
            roi[0] = roi_box
            roi = roi / 255
            prediction = classifier.predict(roi)
            emotion_label = emotion_labels[np.argmax(prediction)]
            label_position = (x, y)
            cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            sock.sendto((emotion_label).encode('utf-8'), server)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
