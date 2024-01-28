from time import sleep

import cv2
from keras.src.saving.saving_api import load_model
from ultralytics import YOLO
from inference.predictor import process_image, predict_image, draw_label

modelYolo = YOLO('../../../models/yolov8n-face.pt')
model_age = load_model('../../../models/model_age_48.model')
model_gen = load_model('../../../models/model_gen_48.model')
model_rac = load_model('../../../models/model_race_48.model')
model_emo = load_model(('../../models/model_pictures_fer_bin.h5'))


image_full = cv2.imread("/home/vorkov/Pictures/Webcam/2023-11-16-190519.jpg")
cv2.imshow("YOLOv8 Tracking", image_full)

# PERSON DETECTION
results = modelYolo.track(image_full)

if len(results[0]) != 0:

    # image cropping
    current_ind = results[0].boxes.cls.int().tolist()[0]
    xyxy = results[0].boxes.xyxy[current_ind].int().tolist()

    #prediction making
    img_gray = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY) # костыль для перевода картинки в чб связан со спецификой используемого датасета
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) # костыль для перевода картинки в чб связан со спецификой используемого датасета
    image_array, image_resized = process_image(img_gray, xyxy)
    pred_age, pred_gen, pred_rac, pred_emo = predict_image(image_array,  model_age, model_gen, model_rac, model_emo)
    image_full = results[0].plot()

    # session_start_time = time.time()
    # label_person1 = "service time: {} sec".format(int(time.time() - session_start_time))
    # draw_label(annotated_frame, (0, annotated_frame.shape[0]-10), label_person1)
    # label_person2 = "client counter: {}".format(client_number)
    # draw_label(annotated_frame, (0, annotated_frame.shape[0]-30), label_person2)
    label_age = "age: {}".format(pred_age)
    print(label_age)
    draw_label(image_full, (0, image_full.shape[0]-50), label_age)
    label_gender = "sex: {}".format(pred_gen)
    print(label_gender)
    draw_label(image_full, (0, image_full.shape[0]-70), label_gender)
    label_race = "race: {}".format(pred_rac)
    print(label_race)
    draw_label(image_full, (0, image_full.shape[0]-90), label_race)
    label_emotion = "20240118-231355: {}".format(pred_emo)
    print(label_emotion)
    draw_label(image_full, (0, image_full.shape[0]-110), label_emotion)

    cv2.imshow("YOLOv8 Full", image_full)
    cv2.imshow("YOLOv8 Resized", image_resized) # дебажный вывод

cv2.waitKey(0)
cv2.destroyAllWindows()
