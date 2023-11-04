import cv2
import numpy as np
import time
import json
import yaml
import dlib
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

from omegaconf import OmegaConf
from pathlib import Path

REAL_FPS = 5
FACE_CLASSIFIER_MIN_NEIGHBORS = 12
FACE_CLASSIFIER_MIN_SIZE = (56, 56)

mutex = Lock()
modelYolo = YOLO('../models/yolov8n.pt')


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def get_model(cfg):
    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_sex = Dense(units=2, activation="softmax", name="pred_sex")(features)
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_sex, pred_age])
    return model


detector = dlib.get_frontal_face_detector()
model_name, img_size = Path("EfficientNetB3_224_weights.11-3.44.hdf5").stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
model = get_model(cfg)
model.load_weights("../models/EfficientNetB3_224_weights.11-3.44.hdf5")


def try_detect_frame(worker_id: int, video_driver_path: str, cap: any, client_number: int):
    print("new detection")
    worker = dict.fromkeys(EMOTION_LABELS, 0)  # для каждого айдишника в словаре задаем словарь эмоций
    worker["worker_id"] = worker_id

    prev = 0
    sex_avg = 0
    sex_count = 0
    age_avg = 0
    age_count = 0

    detected_track_id = -1

    kostyl = 1
    while (True):
        is_frame, image_full = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(cap.get(cv2.CAP_PROP_FPS))
        if int(cap.get(
                cv2.CAP_PROP_FPS) / REAL_FPS) != kostyl:  # программа захватит все кадры и будет тормозить, необходимо отбросить большую часть
            kostyl += 1
            continue
        kostyl = 1
        # print("process time:" + str(time.time() - prev))
        prev = time.time()

        if is_frame == False:
            print("no frame")
            continue

        # image_full = image_full[int(image_full.shape[0]/5) : int(4*image_full.shape[0]/5), int(image_full.shape[1]/5) : int(4*image_full.shape[1]/5)]
        cv2.imshow("YOLOv8 Tracking cropped", image_full)
        # PERSON DETECTION
        results = modelYolo.track(image_full, persist=True)
        print("process time:" + str(time.time() - prev))
        # annotated_frame = results[0].plot()
        annotated_frame = image_full
        if results[0].boxes.id == None:  # если объект появлялся на камере и при этом пропал из списка обнаруженных
            continue

        if detected_track_id == -1:
            person_inds = [i for i, j in enumerate(results[0].boxes.cls.int().tolist()) if
                           j == 0]  # получаем айдишники для объектов == человек в векторе айдишников
            if len(person_inds) == 0:  # нормально, если первым кадром нейронка спутала человека с котом
                continue

            person_ind = person_inds[0]  # пусть детектим первого попавшегося
            detected_track_id = results[0].boxes.id.int().tolist()[
                person_ind]  # достаем айдишник объекта и сохраняем его
            session_start_time = time.time()

        is_required_id_exist = False
        for person_ind, id in enumerate(results[0].boxes.id.int().tolist()):
            if detected_track_id == id:
                is_required_id_exist = True
                break

        if is_required_id_exist == False:
            print("no is_required_id_exist")
            print(results[0].boxes.id.int().tolist())
            print(detected_track_id)
            break

        xyxy = results[0].boxes.xyxy[person_ind].int().tolist()
        image_person = image_full[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

        # HEAD DETECTION
        print("process time:" + str(time.time() - prev))
        detected = detector(image_person, 1)
        print("process time:" + str(time.time() - prev))
        faces = np.empty((1, img_size, img_size, 3))
        img_h, img_w, _ = np.shape(image_person)
        if len(detected) == 0:
            continue

        x1, y1, x2, y2, w, h = detected[0].left(), detected[0].top(), detected[0].right() + 1, detected[0].bottom() + 1, \
        detected[0].width(), detected[0].height()
        xw1 = max(int(x1 - 0.3 * w), 0)
        yw1 = max(int(y1 - 0.3 * h), 0)
        xw2 = min(int(x2 + 0.3 * w), img_w - 1)
        yw2 = min(int(y2 + 0.3 * h), img_h - 1)
        image_head = image_person[yw1:yw2 + 1, xw1:xw2 + 1]
        faces[0] = cv2.resize(image_head, (img_size, img_size))

        # predict ages and sexs of the detected faces
        results = model.predict(faces)
        predicted_sexs = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        predicted_age = predicted_ages[0]
        predicted_sex = predicted_sexs[0][0]
        age_avg = (age_avg * age_count + predicted_age) / (sex_count + 1)
        sex_avg = (sex_avg * sex_count + predicted_sex) / (sex_count + 1)
        age_count += 1
        sex_count += 1

        # FACE DETECTION
        image_face = cv2.cvtColor(image_head, cv2.COLOR_BGR2GRAY)
        image_face = cv2.cvtColor(image_face, cv2.COLOR_GRAY2RGB)
        faces = face_classifier.detectMultiScale(image_face, minNeighbors=FACE_CLASSIFIER_MIN_NEIGHBORS,
                                                 minSize=FACE_CLASSIFIER_MIN_SIZE)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]

        # face cutting
        image_face = image_face[y:y + h, x:x + w]

        # some image processing and kostyls
        image_face_resized = cv2.resize(image_face, (48, 48), interpolation=cv2.INTER_AREA)
        roi = np.empty((1, 48, 48, 3))
        roi[0] = image_face_resized
        roi = roi / 255

        # prediction making
        print("process time:" + str(time.time() - prev))
        prediction = classifier.predict(roi)
        print("process time:" + str(time.time() - prev))
        emotion = EMOTION_LABELS[np.argmax(prediction)]
        worker[emotion] += 1

        label_person1 = "service time: {} sec".format(int(time.time() - session_start_time))
        draw_label(annotated_frame, (0, annotated_frame.shape[0] - 10), label_person1)
        label_person2 = "client counter: {}".format(client_number)
        draw_label(annotated_frame, (0, annotated_frame.shape[0] - 30), label_person2)
        label_head1 = "age: {}".format(int(age_avg))
        draw_label(annotated_frame, (0, annotated_frame.shape[0] - 50), label_head1)
        label_head2 = "sex: {}".format("Male" if sex_avg < 0.5 else "Female")
        draw_label(annotated_frame, (0, annotated_frame.shape[0] - 70), label_head2)
        label_emotion = "emotion: {}".format(emotion)
        draw_label(annotated_frame, (0, annotated_frame.shape[0] - 90), label_emotion)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        print("process time:" + str(time.time() - prev))

    worker["age_group"] = int(age_avg)
    worker["sex"] = bool(sex_avg > 0.5)
    service_time = int(time.time() - session_start_time)
    worker["consultation_time"] = service_time
    worker["date"] = int(datetime.now().timestamp())
    mq_send(json.dumps(worker))
    try_detect_frame(worker_id, video_driver_path, cap, client_number if service_time < 10 else client_number + 1)


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

try_detect_frame(worker_id, video_driver_path, cap, 1)

print("end")
cv2.destroyAllWindows()
