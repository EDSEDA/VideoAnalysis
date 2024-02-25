import datetime
import json
import time
from collections import defaultdict
from statistics import mode, mean

import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from torchvision import transforms
import face_recognition

from api.config import EMOTION_LABELS, EMOTION_LABELS_BIN, GENDER_LABELS, RACES_LABELS
import models
from api.rabbit import mq_send
from storage import *


def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def resize_image(image, left, top, right, bottom):
    print(left, top, right, bottom)

    img_height, img_width, _ = image.shape
    height, width = bottom - top, right - left

    print(width, height)
    SIZE_COEFF = 0.4
    new_left = left-SIZE_COEFF*width if left-SIZE_COEFF*width >= 0 else 0
    new_right = right+SIZE_COEFF*width if right+SIZE_COEFF*width <= img_width else img_width
    new_top = top-SIZE_COEFF*height if top-SIZE_COEFF*height >= 0 else 0
    new_bottom = bottom+SIZE_COEFF*height if bottom+SIZE_COEFF*height <= img_height else img_height

    if height > width:  # если изображение вытянуто по вертикали - нужно добавить ширину
        diff_size = height - width
        diff_left = left - 0  # отступ от края картинки до контура лица слева
        diff_right = image.shape[1] - right  # отступ от края картинки до контура лица справа
        if diff_right > diff_left:
            new_left = left - diff_size / 2 if left > 0 + diff_size / 2 else 0  # новый отступ будер равняться максимально возможному расстоянию, но меньше половины разницы между высотой и шириной
            new_right = right + (diff_size - (left - new_left))
        elif diff_left > diff_right:
            new_right = right + diff_size / 2 if right < image.shape[1] - diff_size / 2 else image.shape[1] - 1
            new_left = left - (diff_size - (new_right - right))
        else:
            new_left = left - diff_size / 2
            new_right = right + diff_size / 2
    else:
        diff_size = width - height
        diff_up = top - 0  # отступ от края картинки до контура лица слева
        diff_down = image.shape[0] - bottom  # отступ от края картинки до контура лица справа
        if diff_down > diff_up:
            new_top = top - diff_size / 2 if top > 0 + diff_size / 2 else 0  # новый отступ будер равняться максимально возможному расстоянию, но меньше половины разницы между высотой и шириной
            new_bottom = bottom + (diff_size - (top - new_top))
        elif diff_up > diff_down:
            new_bottom = bottom + diff_size / 2 if bottom < image.shape[0] - diff_size / 2 else image.shape[0] - 1
            new_top = top - (diff_size - (new_bottom - bottom))
        else:
            new_top = top - diff_size / 2
            new_bottom = bottom + diff_size / 2

    image = image[int(new_top):int(new_bottom), int(new_left):int(new_right)]
    return image

def process_image(image_resized):
    image_resized =  Image.fromarray(image_resized)
    image_processed = transform(image_resized)
    image_processed = image_processed.unsqueeze(0)

    return image_processed

def predict_image(image_processed, model_age, model_gen, model_rac, model_emo):
    with torch.no_grad():
        output = model_age(image_processed)
        age = int(output.item())

        output = model_gen(image_processed)
        gender = int(output.round().item())

        output = model_rac(image_processed)
        _, predicted = torch.max(output, 1)
        race = predicted.item()

        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        image_processed_emo = transform(image_processed)
        output = model_emo(image_processed_emo)
        _, predicted = torch.max(output, 1)
        emotion = predicted.item()

    return age, gender, race, emotion

def identify(image):
    face_encodings = face_recognition.face_encodings(image.copy())

    face_names = []
    for face_encoding in face_encodings:  # Перебираем все эмбеддинги с кадра и ищем в истории похожие эмбединги
        matches = face_recognition.compare_faces(identified_face_encodings, face_encoding, tolerance=0.57)  # За настройку определения похожих эмбедингов отвечает коэффициент tolerance
        name = "Unknown Person"

        if True in matches:
            first_match_index = matches.index(True)
            name = identified_face_names[first_match_index]
        else:  # Если человек не был идентицицирован ранее, то добавляем его в список идентифицированных как нового
            identified_face_encodings.append(face_encoding)
            identified_face_names.append(name)

        face_names.append(name)

    return face_locations, face_encodings, face_names

def find_area(left, top, right, bottom):
    return abs((right - left) * (bottom - top))

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def try_detect_frame(worker_id: int, cap: any):
    while (True):

        start = datetime.datetime.now()
        is_frame, image = cap.read()
        print("time cap:", str((datetime.datetime.now() - start).total_seconds()))

        if is_frame == False:
            print("no frame")
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        # PERSON DETECTION
        image = cv2.resize(image, (0, 0), fx=0.50, fy=0.50)
        facesList = model_face.predict(image)
        print("time face finding:", str((datetime.datetime.now() - start).total_seconds()))

        detected_faces = [] # left, top, right, bottom
        for person in facesList:
            person_coords = person.boxes[0].xyxy.tolist()[0] # todo разобраться когда boxes не имеет значений
            coords = int(person_coords[0]), int(person_coords[1]), int(person_coords[2]), int(person_coords[3])
            detected_faces.append(coords)

        max_area = 0
        face_left, face_top, face_right, face_bottom = 0, 0, 0, 0
        for coords in detected_faces: # Поиск максимального по площади лица
            area = find_area(coords[0], coords[1], coords[2], coords[3])
            if max_area < area:
                max_area = area
                face_left, face_top, face_right, face_bottom = coords[0], coords[1], coords[2], coords[3]

        # Добавление пикселей вокруг изображения до квадрата (потому что обучали на квадратных изображениях)
        image_resized = resize_image(image, face_left, face_top, face_right, face_bottom)
        print("time image_resized:", str((datetime.datetime.now() - start).total_seconds()))

        # Идентификация на обрезанном изображении
        face_locations, face_encodings, face_names = identify(image_resized)
        if len(face_names) != 1: # Не должно быть такого, что на данном этапе не определилиникого на фото. Мы уже прежде убедились что человек там есть
            continue

        face_name = face_names[0]
        print("time ident:", str((datetime.datetime.now() - start).total_seconds()))


        names[last_val_iterator["general"]] = face_name
        last_val_iterator["general"] += 1
        if last_val_iterator["general"] == 10:
            last_val_iterator["general"] = 0
            full_flag["general"] = True

        max_val_name = 10 if full_flag["general"] else last_val_iterator["general"]
        name = mode(names[0:max_val_name])

        # Нормалиация изображения к тому виду на котором обучали
        image_processed = process_image(image_resized)
        print("time image_processed:", str((datetime.datetime.now() - start).total_seconds()))

        # Предсказание визуальных параметров
        pred_age, pred_gen, pred_rac, pred_emo = predict_image(image_processed,  model_age, model_gen, model_rac, model_emo)
        print("time predicted:", str((datetime.datetime.now() - start).total_seconds()))

        age_last_values[name][last_val_iterator[name]] = pred_age
        sex_last_values[name][last_val_iterator[name]] = pred_gen
        race_last_values[name][last_val_iterator[name]] = pred_rac
        emotion_last_values[name][last_val_iterator[name]] = pred_emo
        last_val_iterator[name] += 1
        if last_val_iterator[name] == ACCUMULATION_COUNT:
            last_val_iterator[name] = 0
            full_flag[name] = True

        # Дебажная визуальная информация для отладки
        cv2.rectangle(image, (face_left, face_top), (face_right, face_bottom), color=(230, 230, 230), thickness=1) # Рисуем рамку вокруг лица
        cv2.putText(image, name, (face_left - 30, face_bottom - 4), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1) # Подписываем имя

        max_val = ACCUMULATION_COUNT if full_flag[name] else last_val_iterator[name]

        mean_age = mean(age_last_values[name][0:max_val])
        label_age = "age: {}".format(int(mean_age))
        draw_label(image, (0, image.shape[0]-15), label_age)

        mean_gen = mode(sex_last_values[name][0:max_val])
        label_gender = "sex: {}".format(GENDER_LABELS[int(mean_gen)])
        draw_label(image, (0, image.shape[0]-30), label_gender)

        mean_race = mode(race_last_values[name][0:max_val])
        label_race = "race: {}".format(RACES_LABELS[int(mean_race * 0.5)])
        draw_label(image, (0, image.shape[0]-45), label_race)

        mean_emotion = mode(emotion_last_values[name][0:max_val])
        label_emotion = "emotion: {}".format(EMOTION_LABELS_BIN[int(mean_emotion)])
        draw_label(image, (0, image.shape[0]-60), label_emotion)

        print("time drowing:", str((datetime.datetime.now() - start).total_seconds()))

        worker = dict()
        worker["name"] = kostyName[name]
        worker["carModels"] = carModels[name]
        worker["gasStation"] = gasStation[name]
        worker["indexes"] = indexes[name]
        worker["sails"] = sails[name]
        worker["recommendations"] = recomendations[name]
        mq_send(json.dumps(worker))

        end = datetime.datetime.now()
        label_fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        draw_label(image, (0, image.shape[0] - 75), label_fps)

        image = cv2.resize(image, (0, 0), fx=2, fy=2)
        cv2.imshow("YOLOv8 Tracking", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACCUMULATION_COUNT= 50
SIZE = 48
transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Models initialization:")
model_person = YOLO('../models/yolov8n.pt')
model_person.to("cuda")
print("Person model initialized")

model_face = YOLO('../models/yolov8n-face.pt')
model_person.to("cuda")
print("Face model initialized")

# tracker = DeepSort(max_age=50) ## ToDo начала падать. нужно разобраться почему
print("DeepSort tracker have initialized")

model_age = models.AgeEstimatorModel()
model_age.load_state_dict(torch.load("../models/age_model_weights.pth", map_location=device))
model_age.eval()
print("Age model have initialized")

model_gen = models.SexEstimatorModel()
model_gen.load_state_dict(torch.load("../models/sex_model_weights.pth", map_location=device))
model_gen.eval()
print("Gender model have initialized")

model_rac = models.RaceEstimatorModel(5)
model_rac.load_state_dict(torch.load("../models/race_model_weights.pth", map_location=device))
model_rac.eval()
print("Race model have initialized")

model_emo = models.EmotionBinEstimatorModel(3)
model_emo.load_state_dict(torch.load("../models/emotion_bin_model_weights.pth", map_location=device))
model_emo.eval()
print("Emotion model have initialized")

