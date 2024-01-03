import time

import numpy as np
from keras.preprocessing import image
from keras.src.saving.saving_api import load_model
from ultralytics import YOLO
import cv2
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort


from api.config import EMOTION_LABELS, EMOTION_LABELS_BIN, GENDER_LABELS, RACES_LABELS

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def process_image(image_full, xyxy):
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
            new_x_left = xyxy[0] - diff_size / 2 if xyxy[0] > 0 + diff_size / 2 else 0  # новый отступ будер равняться максимально возможному расстоянию, но меньше половины разницы между высотой и шириной
            new_x_right = xyxy[2] + (diff_size - (xyxy[0] - new_x_left))
        elif diff_left > diff_right:
            new_x_right = xyxy[2] + diff_size / 2 if xyxy[2] < image_full.shape[1] - diff_size / 2 else image_full.shape[1] - 1
            new_x_left = xyxy[0] - (diff_size - (new_x_right - xyxy[2]))
        else:
            new_x_left = xyxy[0] - diff_size / 2
            new_x_right = xyxy[2] + diff_size / 2
    else:
        diff_size = width - height
        diff_up = xyxy[1] - 0  # отступ от края картинки до контура лица слева
        diff_down = image_full.shape[0] - xyxy[3]  # отступ от края картинки до контура лица справа
        if diff_down > diff_up:
            new_y_up = xyxy[1] - diff_size / 2 if xyxy[1] > 0 + diff_size / 2 else 0  # новый отступ будер равняться максимально возможному расстоянию, но меньше половины разницы между высотой и шириной
            new_y_down = xyxy[3] + (diff_size - (xyxy[1] - new_y_up))
        elif diff_up > diff_down:
            new_y_down = xyxy[3] + diff_size / 2 if xyxy[3] < image_full.shape[0] - diff_size / 2 else image_full.shape[0] - 1
            new_y_up = xyxy[1] - (diff_size - (new_y_down - xyxy[3]))
        else:
            new_y_up = xyxy[1] - diff_size / 2
            new_y_down = xyxy[3] + diff_size / 2

    image_cropped = image_full[int(new_y_up):int(new_y_down), int(new_x_left):int(new_x_right)]
    # print(image_cropped.shape)
    # print(image_cropped)
    # print("image_cropped")
    # cv2.imwrite("/home/vorkov/Workspace/EDA/learning/data/trash/me1.jpg", image_cropped)
    # image_cropped = cv2.imread("/home/vorkov/Workspace/EDA/learning/data/trash/me1.jpg")
    # print(image_cropped.shape)
    # print(image_cropped)


    # print(image_full.shape)
    # print(image_cropped.shape)

    SIZE = 48
    image_cropped = cv2.GaussianBlur(image_cropped, (3, 3), 0)
    image_resized = cv2.resize(image_cropped, (SIZE, SIZE))  # Если необходимо изменить размер изображения
    # image_array = np.empty((1, SIZE, SIZE, 3))
    # image_array[0] = image_resized
    # image_array /= 255.0

    img_array = image.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = to_grayscale_then_rgb(img_array)  # Примените вашу предварительную обработку
    img_array /= 255.0  # Приведение к диапазону [0, 1]
    # predictions = model_emo.predict(img_array)
    # predicted_class = np.argmax(predictions, axis=1)
    # print("AAAAA" + str(predicted_class))

    # string = "/home/vorkov/Workspace/EDA/learning/data/test/test"+str(time.time())+".jpg"
    # cv2.imwrite(str(string), image_resized)
    return img_array, image_resized

def predict_image(image_array, model_age, model_gen, model_rac, model_emo):
    age = int(model_age.predict(image_array))
    gender = GENDER_LABELS[np.round(np.argmax(model_gen.predict(image_array)))]
    race = RACES_LABELS[np.round(np.argmax(model_rac.predict(image_array)))]
    # emotion = EMOTION_LABELS[np.round(np.argmax(model_emo.predict(image_array)))]
    emo_arr = model_emo.predict(image_array)
    print(emo_arr)
    emotion = EMOTION_LABELS_BIN[np.round(np.argmax(emo_arr))]

    return age, gender, race, emotion

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def try_detect_frame(worker_id: int, video_driver_path: str, cap: any, client_number: int):
    while (True):
        is_frame, image_full = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(cap.get(cv2.CAP_PROP_FPS))

        if is_frame == False:
            print("no frame")
            continue

        # PERSON DETECTION
        results = modelYolo.track(image_full)
        if len(results[0]) != 0:

            # image cropping
            current_ind = results[0].boxes.cls.int().tolist()[0]
            xyxy = results[0].boxes.xyxy[current_ind].int().tolist()

            #prediction making
            image_array, image_resized = process_image(image_full, xyxy)
            pred_age, pred_gen, pred_rac, pred_emo = predict_image(image_array,  model_age, model_gen, model_rac, model_emo)
            image_full = results[0].plot()

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

        cv2.imshow("YOLOv8 Tracking", image_full)
        cv2.imshow("YOLOv8 Resized", image_resized) # дебажный вывод

modelYolo = YOLO('../models/yolov8n-face.pt')
tracker = DeepSort(max_age=50)
model_age = load_model('../models/model_age_48.model')
model_gen = load_model('../models/model_gen_48.model')
model_rac = load_model('../models/model_race_48.model')
model_emo = load_model('../models/model_pictures_fer_bin.h5')
