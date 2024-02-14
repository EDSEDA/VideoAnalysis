import datetime
import json
import time
from collections import defaultdict

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
        gender = GENDER_LABELS[int(output.round().item())]

        output = model_rac(image_processed)
        _, predicted = torch.max(output, 1)
        race = RACES_LABELS[predicted.item()]

        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        image_processed_emo = transform(image_processed)
        output = model_emo(image_processed_emo)
        _, predicted = torch.max(output, 1)
        emotion = EMOTION_LABELS_BIN[predicted.item()]

    return age, gender, race, emotion

def identify(image):
    # ToDo гспд уберите кто-нибудь этот костыль
    cv2.imwrite("image.jpeg", image)
    image = face_recognition.load_image_file("image.jpeg")

    face_locations = face_recognition.face_locations(image)
    print("face locations: ", str(face_locations))

    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names = []
    for face_encoding in face_encodings:  # Перебираем все эмбеддинги с кадра и ищем в истории похожие эмбединги
        matches = face_recognition.compare_faces(identified_face_encodings, face_encoding, tolerance=0.66)  # За настройку определения похожих эмбедингов отвечает коэффициент tolerance
        name = "Unknown Person"

        if True in matches:
            first_match_index = matches.index(True)
            name = identified_face_names[first_match_index]
        else:  # Если человек не был идентицицирован ранее, то добавляем его в список идентифицированных
            identified_face_encodings.append(face_encoding)
            identified_face_names.append(name)

        face_names.append(name)

    return face_locations, face_encodings, face_names

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.2, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def try_detect_frame(worker_id: int, cap: any):
    while (True):

        start = datetime.datetime.now()
        is_frame, image = cap.read()
        print("time cap:", str((datetime.datetime.now() - start).total_seconds()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if is_frame == False:
            print("no frame")
            continue

        # PERSON DETECTION
        # personsList = model_person.predict(image) # Детектим людей целиком чтобы не терять их при отворачивании лица
        # print("time person:", str((datetime.datetime.now() - start).total_seconds()))

        facesList = model_face.predict(image) # Детектим людей целиком чтобы не терять их при отворачивании лица
        print("time face:", str((datetime.datetime.now() - start).total_seconds()))
        for i, person in enumerate(facesList):
            person_coords = person.boxes[0].xyxy.tolist()[0]
            left, top, right, bottom = int(person_coords[0]), int(person_coords[1]), int(person_coords[2]), int(person_coords[3])
            print(left, top, right, bottom)
            # Добавление пикселей вокруг изображения до квадрата (потому что обучали на квадратных изображениях)
            image_resized = resize_image(image, left, top, right, bottom)
            print("time image_resized:", str((datetime.datetime.now() - start).total_seconds()))

            # Идентификация на обрезанном изображении
            face_locations, face_encodings, face_names = identify(image_resized)
            if len(face_names) == 0:
                continue
            name = face_names[0]
            print("time ident:", str((datetime.datetime.now() - start).total_seconds()))

            # Нормалиация изображения к тому виду на котором обучали
            image_processed = process_image(image_resized)
            print("time image_processed:", str((datetime.datetime.now() - start).total_seconds()))

            # Предсказание визуальных параметров
            pred_age, pred_gen, pred_rac, pred_emo = predict_image(image_processed,  model_age, model_gen, model_rac, model_emo)
            print("time predicted:", str((datetime.datetime.now() - start).total_seconds()))

            # Дебажная визуальная информация для отладки
            cv2.rectangle(image, (left, top), (right, bottom), color=(230, 230, 230), thickness=10) # Рисуем рамку вокруг лица
            cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 2) # Подписываем имя

            label_age = "age: {}".format(pred_age)
            draw_label(image, (i*400, image.shape[0]-30), label_age)
            label_gender = "sex: {}".format(pred_gen)
            draw_label(image, (i*400, image.shape[0]-60), label_gender)
            label_race = "race: {}".format(pred_rac)
            draw_label(image, (i*400, image.shape[0]-90), label_race)
            label_emotion = "emotion: {}".format(pred_emo)
            draw_label(image, (i*400, image.shape[0]-120), label_emotion)
            print("time drowing:", str((datetime.datetime.now() - start).total_seconds()))

            worker = dict()
            worker["name"] = name
            worker["carModels"] = carModels[name]
            worker["gasStation"] = gasStation[name]
            worker["indexes"] = indexes[name]
            worker["sails"] = sails[name]
            worker["recomendations"] = recomendations[name]
            mq_send(json.dumps(worker))

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(image, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.imshow("YOLOv8 Tracking", image)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

identified_face_encodings = []
identified_face_names = []

name = "Kostylev Ivan"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person1.1.jpg")
face_locations = face_recognition.face_locations(learned_person)
learned_person_encoding = face_recognition.face_encodings(learned_person, face_locations)[0]
identified_face_encodings.append(learned_person_encoding)
identified_face_names.append(name)

name = "Vorkov Nikita"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person2.1.jpg")
face_locations = face_recognition.face_locations(learned_person)
learned_person_encoding = face_recognition.face_encodings(learned_person, face_locations)[0]
identified_face_encodings.append(learned_person_encoding)
identified_face_names.append(name)

name = "Bondarenko Andrey"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person2.1.jpg")
face_locations = face_recognition.face_locations(learned_person)
learned_person_encoding = face_recognition.face_encodings(learned_person, face_locations)[0]
identified_face_encodings.append(learned_person_encoding)
identified_face_names.append(name)

name = "Sulimenko Nikita"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person4.1.jpg")
face_locations = face_recognition.face_locations(learned_person)
learned_person_encoding = face_recognition.face_encodings(learned_person, face_locations)[0]
identified_face_encodings.append(learned_person_encoding)
identified_face_names.append(name)

name = "Popov Alexander"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person5.1.jpg")
face_locations = face_recognition.face_locations(learned_person)
learned_person_encoding = face_recognition.face_encodings(learned_person, face_locations)[0]
identified_face_encodings.append(learned_person_encoding)
identified_face_names.append(name)

name = "Karatetskaya Maria"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person6.1.jpg")
face_locations = face_recognition.face_locations(learned_person)
learned_person_encoding = face_recognition.face_encodings(learned_person, face_locations)[0]
identified_face_encodings.append(learned_person_encoding)
identified_face_names.append(name)


recomendations = dict()
recomendations["Kostylev Ivan"] = ["шоколад", "вода 0.5л",  "яйца",  "сок 1л",  "сухарики",  "мороженое" ]
recomendations["Vorkov Nikita"] = ["мороженое", "сок 1л",  "яйца", "леденец",  "сухарики", "чипсы"  ]
recomendations["Bondarenko Andrey"] = ["яйца", "сок 1л",  "яйца",  "мороженое",  "чипсы",  "мороженое" ]
recomendations["Sulimenko Nikita"] = ["мандарины", "леденец",  "яйца",  "чипсы",  "сухарики",  "чипсы" ]
recomendations["Popov Alexander"] = ["леденец", "шоколад",  "яйца",  "сок 1л",  "сухарики",  "мороженое" ]
recomendations["Karatetskaya Maria"] = ["шоколад", "вода 0.5л",  "яйца",  "сок 1л",  "сухарики",  "мороженое" ]
recomendations["Unknown Person"] = ["шоколад", "яйца",  "мороженое",  "сухарики",  "леденец",  "чипсы" ]

carModels = dict()
carModels["Kostylev Ivan"] = "Lada Priora"
carModels["Vorkov Nikita"] = "Tesly CyperTrack"
carModels["Bondarenko Andrey"] = "Tayoty Mark2"
carModels["Sulimenko Nikita"] = "BMW X5"
carModels["Popov Alexander"] = "Volkswagen Golf"
carModels["Karatetskaya Maria"] = "without car"
carModels["Unknown Person"] = "Porche 911"

gasStation = dict()
gasStation["Kostylev Ivan"] = 1
gasStation["Vorkov Nikita"] = 2
gasStation["Bondarenko Andrey"] = 3
gasStation["Sulimenko Nikita"] = 4
gasStation["Popov Alexander"] = 5
gasStation["Karatetskaya Maria"] = 0
gasStation["Unknown Person"] = 0

indexes = dict()
indexes["Kostylev Ivan"] = 13
indexes["Vorkov Nikita"] = 2024
indexes["Bondarenko Andrey"] = 69
indexes["Sulimenko Nikita"] = 21
indexes["Popov Alexander"] = 100
indexes["Karatetskaya Maria"] = 7
indexes["Unknown Person"] = 7

sails = dict()
sails["Kostylev Ivan"] = 5
sails["Vorkov Nikita"] = 10
sails["Bondarenko Andrey"] = 25
sails["Sulimenko Nikita"] = 50
sails["Popov Alexander"] = 30
sails["Karatetskaya Maria"] = 15
sails["Unknown Person"] = 15

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []