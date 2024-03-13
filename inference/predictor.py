import datetime
import json
import time
from collections import defaultdict
from statistics import mode, mean

from PIL import Image
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition

import cv2
from init import *

from grifon.video_analysis.schema import VideoAnalysMessage
from grifon.video_analysis.enums import SexEnum
from grifon.video_analysis.enums import RaceEnum
from grifon.video_analysis.enums import EmotionEnum

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def resize_image(image, left, top, right, bottom):
    img_height, img_width, _ = image.shape
    height, width = bottom - top, right - left

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
    face_embeddings = face_recognition.face_encodings(image.copy()) # тут происходит неявное запоминание лиц либой, если они прежде не были нам известны

    person_ids = []
    for face_embedding in face_embeddings:  # Перебираем все эмбеддинги с кадра и ищем в истории похожие эмбединги
        matches = face_recognition.compare_faces(identified_face_embeddings, face_embedding, tolerance=0.57)  # За настройку определения похожих эмбедингов отвечает коэффициент tolerance
        name_id = int(time.time() * 10e6)

        if True in matches:
            first_match_index = matches.index(True)
            name_id = identified_person_ids[first_match_index]
        else:  # Если человек не был идентицицирован ранее, то добавляем его в список идентифицированных как нового
            identified_face_embeddings.append(face_embedding)
            identified_person_ids.append(name_id)

        person_ids.append(name_id)

    return face_embeddings, person_ids

def find_area(left, top, right, bottom):
    return abs((right - left) * (bottom - top))

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def try_detect_frame(cap):
    global identified_person_frames_counter
    global analyzed_person_frames_counter
    global last_identified_person_id

    while (True):

        # ПРИЕМ ВИДЕО
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        start = datetime.datetime.now()
        is_frame, image = cap.read()

        if is_frame == False:
            print("no frame")
            continue



        # ДЕТКЦИЯ
        image = cv2.resize(image, (0, 0), fx=0.50, fy=0.50) # ужимаем изображение чтобы быстрее его обрабатывать
        facesList = model_face.predict(image) # ищем людей на изображении

        if len(facesList) == 0: # если из кадра пропали все, то сбрасываем текущего человека
            identified_person_frames_counter = 0
            cv2.imshow("YOLOv8 Tracking", image)
            continue

        detected_faces = [] # left, top, right, bottom
        for person in facesList:
            if len (person.boxes) != 0:
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



        # ИДЕНТИФИКАЦИЯ
        face_embeddings, person_ids = identify(image_resized)
        if len(face_embeddings) != 1: # Не должно быть такого, что на данном этапе не определилиникого на фото. Мы уже прежде убедились что человек там есть
            continue
        current_person_id = person_ids[0] # Даже если на камеру попало 2 лица, то берем просто первое из них

        identified_person_ids[identified_person_frames_counter % identified_required_person_frames] = current_person_id
        identified_person_frames_count = identified_required_person_frames if (identified_person_frames_counter // identified_required_person_frames != 0) else identified_person_frames_counter % identified_required_person_frames
        mode_person_id = mode(identified_person_ids[0:identified_person_frames_count+1]) # Получаем моду из айдишников последних n эмбеддингов

        if mode_person_id != last_identified_person_id: # Если айди лица новый, то сбрасываем счетчик фреймов для прошлого человека
            analyzed_person_frames_counter = 0



        # АНАЛИЗ ВНЕШНИХ ПРИЗНАКОВ
        image_processed = process_image(image_resized) # Нормалиация изображения к тому виду на котором обучали
        pred_age, pred_sex, pred_rac, pred_emo = predict_image(image_processed,  model_age, model_gen, model_rac, model_emo) # Предсказание визуальных параметров

        age_last_values[analyzed_person_frames_counter % analyzed_required_person_frames] = pred_age
        sex_last_values[analyzed_person_frames_counter % analyzed_required_person_frames] = pred_sex
        rac_last_values[analyzed_person_frames_counter % analyzed_required_person_frames] = pred_rac
        emo_last_values[analyzed_person_frames_counter % analyzed_required_person_frames] = pred_emo
        analyzed_person_frames_count = analyzed_required_person_frames if (analyzed_person_frames_counter // analyzed_required_person_frames != 0) else analyzed_person_frames_counter % analyzed_required_person_frames
        mean_age = mean(age_last_values[0:analyzed_person_frames_count + 1]) # Получаем моду из айдишников последних n эмбеддингов
        mean_sex = mode(sex_last_values[0:analyzed_person_frames_count + 1])
        mean_rac = mode(rac_last_values[0:analyzed_person_frames_count + 1])
        mean_emo = mode(emo_last_values[0:analyzed_person_frames_count + 1])

        identified_person_frames_counter += 1
        analyzed_person_frames_counter += 1
        last_identified_person_id = current_person_id



        # ВИЗУАЛИЗАЦИЯ (дебажная)
        label_age = "age: {}".format(int(mean_age))
        draw_label(image, (0, image.shape[0]-15), label_age)
        label_gender = "sex: {}".format(GENDER_LABELS[int(mean_sex)])
        draw_label(image, (0, image.shape[0]-30), label_gender)
        label_race = "race: {}".format(RACES_LABELS[int(mean_rac)])
        draw_label(image, (0, image.shape[0]-45), label_race)
        label_emotion = "emotion: {}".format(EMOTION_LABELS_BIN[int(mean_emo)])
        draw_label(image, (0, image.shape[0]-60), label_emotion)
        label_person_id = "person id: {}".format(mode_person_id)
        draw_label(image, (0, image.shape[0]-75), label_person_id)
        label_proc_time = "proc time ms: {}".format(int((datetime.datetime.now() - start).total_seconds()*1000))
        draw_label(image, (0, image.shape[0]-90), label_proc_time)
        cv2.rectangle(image, (face_left, face_top), (face_right, face_bottom), color=(230, 230, 230), thickness=1) # Рисуем рамку вокруг лица
        image = cv2.resize(image, (0, 0), fx=2, fy=2)
        cv2.imshow("YOLOv8 Tracking", image)



        # ОТПРАВКА СООБЩЕНИЯ

        message = VideoAnalysMessage(
            embedding = face_embeddings[0].tolist(),
            person_id = mode_person_id,
            age = int(mean_age),
            sex = SexEnum(GENDER_LABELS[int(mean_sex)]),
            race = RaceEnum(RACES_LABELS[int(mean_rac)]),
            emotion = EmotionEnum(EMOTION_LABELS_BIN[int(mean_emo)]),
        )
        kafka_client.send_message(message.json())

    cap.release()
    cv2.destroyAllWindows()