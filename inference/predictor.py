import datetime
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

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def resize_image(image_full, xyxy):
    new_x_left = xyxy[0]
    new_y_up = xyxy[1]
    new_x_right = xyxy[2]
    new_y_down = xyxy[3]
    width = new_x_right - new_x_left
    height = new_y_down - new_y_up

    if height > width:  # если изображение вытянуто по вертикали - нужно добавить ширину
        diff_size = height - width
        diff_left = xyxy[0] - 0  # отступ от края картинки до контура лица слева
        diff_right = image_full.shape[1] - xyxy[2]  # отступ от края картинки до контура лица справа
        if diff_right > diff_left:
            new_x_left = xyxy[0] - diff_size / 2 if xyxy[
                                                        0] > 0 + diff_size / 2 else 0  # новый отступ будер равняться максимально возможному расстоянию, но меньше половины разницы между высотой и шириной
            new_x_right = xyxy[2] + (diff_size - (xyxy[0] - new_x_left))
        elif diff_left > diff_right:
            new_x_right = xyxy[2] + diff_size / 2 if xyxy[2] < image_full.shape[1] - diff_size / 2 else \
            image_full.shape[1] - 1
            new_x_left = xyxy[0] - (diff_size - (new_x_right - xyxy[2]))
        else:
            new_x_left = xyxy[0] - diff_size / 2
            new_x_right = xyxy[2] + diff_size / 2
    else:
        diff_size = width - height
        diff_up = xyxy[1] - 0  # отступ от края картинки до контура лица слева
        diff_down = image_full.shape[0] - xyxy[3]  # отступ от края картинки до контура лица справа
        if diff_down > diff_up:
            new_y_up = xyxy[1] - diff_size / 2 if xyxy[
                                                      1] > 0 + diff_size / 2 else 0  # новый отступ будер равняться максимально возможному расстоянию, но меньше половины разницы между высотой и шириной
            new_y_down = xyxy[3] + (diff_size - (xyxy[1] - new_y_up))
        elif diff_up > diff_down:
            new_y_down = xyxy[3] + diff_size / 2 if xyxy[3] < image_full.shape[0] - diff_size / 2 else image_full.shape[
                                                                                                           0] - 1
            new_y_up = xyxy[1] - (diff_size - (new_y_down - xyxy[3]))
        else:
            new_y_up = xyxy[1] - diff_size / 2
            new_y_down = xyxy[3] + diff_size / 2

    image_cropped = image_full[int(new_y_up):int(new_y_down), int(new_x_left):int(new_x_right)]
    return image_cropped

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

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def try_detect_frame(worker_id: int, cap: any):
    track_history = defaultdict(lambda: [])
    while (True):

        start = datetime.datetime.now()
        is_frame, image_full = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if is_frame == False:
            print("no frame")
            continue

        # PERSON DETECTION
        results = modelYolo.track(image_full, persist=True, tracker="../cfg/bytetrack.yaml")
        if len(results) != 0 and results[0].boxes.id != None:

            boxes0 = results[0].boxes.xywh.cpu()
            track_ids0 = results[0].boxes.id.int().cpu().tolist()

            image_full = results[0].plot()

            for (x, y, w, h), track_id in zip(boxes0, track_ids0):
                track_history[track_id].append((float(x), float(y)))
                if len(track_history[track_id]) > 20:
                    track_history[track_id].pop(0)

                points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(image_full, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                print(points)

            # Идентификация
            face_locations = face_recognition.face_locations(image_full)
            face_encodings = face_recognition.face_encodings(image_full, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(identified_face_encodings, face_encoding)
                name = "Unknown"

                # Если хотим брать только идентифицированные лица
                if True in matches:
                    first_match_index = matches.index(True)
                    name = identified_face_names[first_match_index]
                    face_names.append(name)

            print(face_names)
            print(face_locations)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                print(name, top, right, bottom, left)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image_full, name, (left + 6, bottom - 6), font, 2.0, (255, 0, 0), 2)

            # image cropping
            current_ind = results[0].boxes.cls.int().tolist()[0]
            xyxy = results[0].boxes.xyxy[current_ind].int().tolist()

            #prediction making
            image_resized = resize_image(image_full, xyxy)
            image_processed = process_image(image_resized)
            pred_age, pred_gen, pred_rac, pred_emo = predict_image(image_processed,  model_age, model_gen, model_rac, model_emo)

            # session_start_time = time.time()
            # label_person1 = "service time: {} sec".format(int(time.time() - session_start_time))
            # draw_label(annotated_frame, (0, annotated_frame.shape[0]-10), label_person1)
            # label_person2 = "client counter: {}".format(client_number)
            # draw_label(annotated_frame, (0, annotated_frame.shape[0]-30), label_person2)
            label_age = "age: {}".format(pred_age)
            draw_label(image_full, (0, image_full.shape[0]-50), label_age)
            label_gender = "sex: {}".format(pred_gen)
            draw_label(image_full, (0, image_full.shape[0]-70), label_gender)
            label_race = "race: {}".format(pred_rac)
            draw_label(image_full, (0, image_full.shape[0]-90), label_race)
            label_emotion = "emotion: {}".format(pred_emo)
            draw_label(image_full, (0, image_full.shape[0]-110), label_emotion)

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(image_full, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.imshow("YOLOv8 Tracking", image_full)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SIZE = 48
transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

modelYolo = YOLO('../models/yolov8n-face.pt')
tracker = DeepSort(max_age=50)

model_age = models.AgeEstimatorModel()
model_age.load_state_dict(torch.load("../models/age_model_weights.pth"))
model_age.eval()

model_gen = models.SexEstimatorModel()
model_gen.load_state_dict(torch.load("../models/sex_model_weights.pth"))
model_gen.eval()

model_rac = models.RaceEstimatorModel(5)
model_rac.load_state_dict(torch.load("../models/race_model_weights.pth"))
model_rac.eval()

model_emo = models.EmotionBinEstimatorModel(3)
model_emo.load_state_dict(torch.load("../models/emotion_bin_model_weights.pth"))
model_emo.eval()

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

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []