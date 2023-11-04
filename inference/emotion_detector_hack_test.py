
import cv2
import numpy as np
import time
import json
import yaml
import dlib

from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from keras.models import load_model
from datetime import datetime
from threading import Thread, Lock
from api.rabbit import mq_send
from api.config import EMOTION_LABELS, paths
from ultralytics import YOLO

from omegaconf import OmegaConf
from pathlib import Path

FRAME_RATE = 10
FACE_CLASSIFIER_MIN_NEIGHBORS=12
FACE_CLASSIFIER_MIN_SIZE=(56, 56)

mutex = Lock()
modelYolo = YOLO('../learning/data/yolov8n.pt')

def try_detect_frame(worker_id: int, video_driver_path: str, cap: any):
    print("new detection")
    session_start_time = time.time()
    worker = dict.fromkeys(EMOTION_LABELS, 0)# для каждого айдишника в словаре задаем словарь эмоций
    worker["worker_id"] = worker_id

    prev = 0
    sex_avg = 0
    sex_count = 0
    age_avg = 0
    age_count = 0

    detected_track_id = -1

    while (True):
        time_elapsed = time.time() - prev
        is_frame, image_full = cap.read()

        if time_elapsed > 1. / FRAME_RATE:

            # if is_frame == False:
            #     print("no frame")
            #     continue
            #
            # prev = time.time()
            #
            # # Трекинг объектов
            # results = modelYolo.track(image_full, persist=True)
            # annotated_frame = results[0].plot()
            # print(results[0].boxes.cls)
            # print(results[0].boxes.id)
            #
            # if results[0].boxes.id == None and detected_track_id == -1:  # если объект никогда не появлялся на камере
            #     continue
            #
            # if results[0].boxes.id == None:  # если объект появлялся на камере и при этом пропал из списка обнаруженных
            #     break
            #
            # if detected_track_id == -1:
            #     person_inds = [i for i, j in enumerate(results[0].boxes.cls.int().tolist()) if j == 0]  # получаем айдишники для объектов == человек в векторе айдишников
            #     if len(person_inds) == 0: # нормально, если первым кадром нейронка спутала человека с котом
            #         continue
            #
            #     person_ind = person_inds[0]  # пусть детектим первого попавшегося
            #     detected_track_id = results[0].boxes.id.int().tolist()[person_ind]  # достаем айдишник объекта и сохраняем его
            #
            # is_required_id_exist = False
            # for id in results[0].boxes.id.int().tolist():
            #     if detected_track_id == id:
            #         is_required_id_exist = True
            #
            # if is_required_id_exist == False:
            #     break
            #
            #
            # xyxy = results[0].boxes.xyxy[person_ind].int().tolist()
            # image_person = image_full[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            # cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # # faces searching
            # image_person = cv2.cvtColor(image_person, cv2.COLOR_BGR2GRAY)
            # image_person = cv2.cvtColor(image_person, cv2.COLOR_GRAY2RGB)
            # faces = face_classifier.detectMultiScale(image_person, minNeighbors=FACE_CLASSIFIER_MIN_NEIGHBORS, minSize=FACE_CLASSIFIER_MIN_SIZE)
            #
            # if len(faces) == 0:
            #     continue
            #
            # x, y, w, h = faces[0]
            #
            # # face cutting
            # image_face = image_person[y:y + h, x:x + w]
            #
            # # some image processing and kostyls
            # image_face_resized = cv2.resize(image_face, (48, 48), interpolation=cv2.INTER_AREA)
            # roi = np.empty((1, 48, 48, 3))
            # roi[0] = image_face_resized
            # roi = roi / 255
            #
            # #prediction making
            # prediction = classifier.predict(roi)
            # emotion_label = EMOTION_LABELS[np.argmax(prediction)]
            # worker[emotion_label] += 1
            #
            # # prediction drawing
            # label_position = (x, y)
            # cv2.putText(image_face, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # cv2.imshow('Emotion Detector', image_face)

            # Importing Models and set mean values
            face1 = "../learning/data/opencv_face_detector.pbtxt"
            face2 = "../learning/data/opencv_face_detector_uint8.pb"
            age1 = "../learning/data/age_deploy.prototxt"
            age2 = "../learning/data/age_net.caffemodel"
            gen1 = "../learning/data/gender_deploy.prototxt"
            gen2 = "../learning/data/gender_net.caffemodel"

            MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

            # Using models
            # Face
            face = cv2.dnn.readNet(face2, face1)

            # age
            age = cv2.dnn.readNet(age2, age1)

            # gender
            gen = cv2.dnn.readNet(gen2, gen1)

            # Categories of distribution
            la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                  '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            lg = ['Male', 'Female']

            # Copy image
            fr_cv = image_full.copy()

            # Face detection
            fr_h = fr_cv.shape[0]
            fr_w = fr_cv.shape[1]
            blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300),
                                         [104, 117, 123], True, False)

            face.setInput(blob)
            detections = face.forward()

            # Face bounding box creation
            faceBoxes = []
            for i in range(detections.shape[2]):

                # Bounding box creation if confidence > 0.7
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    x1 = int(detections[0, 0, i, 3] * fr_w)
                    y1 = int(detections[0, 0, i, 4] * fr_h)
                    x2 = int(detections[0, 0, i, 5] * fr_w)
                    y2 = int(detections[0, 0, i, 6] * fr_h)

                    faceBoxes.append([x1, y1, x2, y2])

                    cv2.rectangle(fr_cv, (x1, y1), (x2, y2), (0, 255, 0), int(round(fr_h / 150)), 8)

            # Checking if face detected or not
            if not faceBoxes:
                print("No face detected")

            # Final results (otherwise)
            # Loop for all the faces detected
            for faceBox in faceBoxes:
                # Extracting face as per the faceBox
                face = fr_cv[max(0, faceBox[1] - 15):
                             min(faceBox[3] + 15, fr_cv.shape[0] - 1),
                       max(0, faceBox[0] - 15):min(faceBox[2] + 15,
                                                   fr_cv.shape[1] - 1)]

                # Extracting the main blob part
                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Prediction of gender
                gen.setInput(blob)
                genderPreds = gen.forward()
                gender = lg[genderPreds[0].argmax()]

                # Prediction of age
                age.setInput(blob)
                agePreds = age.forward()
                age = la[agePreds[0].argmax()]

                # Putting text of age and gender
                # At the top of box
                cv2.putText(fr_cv,
                            f'{gender}, {age}',
                            (faceBox[0] - 150, faceBox[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (217, 0, 0),
                            4,
                            cv2.LINE_AA)

                cv2.imshow('age_sex_detection', fr_cv)

            # # draw results
            # predicted_age = predicted_ages[0]
            # predicted_sex = predicted_sexs[0][0]
            # age_avg = (age_avg * age_count + predicted_age) / (sex_count + 1)
            # sex_avg = (sex_avg * sex_count + predicted_sex) / (sex_count + 1)
            # age_count += 1
            # sex_count += 1
            #
            # label = "ages: {}, sex: {}".format(int(age_avg), "Male" if sex_avg < 0.5 else "Female")
            # draw_label(croped, (detected[0].left(), detected[0].top()), label)
            #
            # cv2.imshow('age_sex_detection', croped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # worker["age_group"] = int(age_avg)
    # worker["sex"] = bool(sex_avg > 0.5)
    # worker["consultation_time"] = int(time.time() - session_start_time)
    # worker["date"] = int(datetime.now().timestamp())
    # mq_send(json.dumps(worker))
    try_detect_frame(worker_id, video_driver_path, cap)



face_classifier = cv2.CascadeClassifier(paths.FACE_CLASSIFIER_PATH)  # детектор лица OpenCV
classifier = load_model(paths.PREDICTION_MODEL_PATH)  # обученная модель для классификации эмоций

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
video_driver_path = worker['video_driver_path']
cap = cv2.VideoCapture(video_driver_path)
try_detect_frame(worker_id, video_driver_path, cap)

print("end")
cv2.destroyAllWindows()
