import torch # ЭТОТ ИМПОРТ ОБЯЗАН БЫТЬ ПЕРВОЙ СТРОЧКОЙ ПРОГРАММЫ, ЕСЛИ ЕГО ПЕРЕСТАВИТЬ ТО ВСЕ ОБЯЗАТЕЛЬНО УПАДЕТ, А ТЕБЕ ОТОРВУТ РУКИ!!!
import cv2
import yaml
import sys
sys.path.append('..')

# from connection import connection
import predictor

CONFIG_PATH: str = "../cfg/emotion_detector.yaml"

# Чтение конфига
with open(CONFIG_PATH, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

gst_stream = config["gst_stream"]
kafka_addr = config["kafka_addr"]
kafka_topic = config["kafka_topic"]

cap = cv2.VideoCapture(gst_stream)
predictor.try_detect_frame(cap)
cap.release()
cv2.destroyAllWindows()

print("cv detector was closed")
