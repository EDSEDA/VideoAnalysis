import torch # ЭТОТ ИМПОРТ ОБЯЗАН БЫТЬ ПЕРВОЙ СТРОЧКОЙ ПРОГРАММЫ, ЕСЛИ ЕГО ПЕРЕСТАВИТЬ ТО ВСЕ ОБЯЗАТЕЛЬНО УПАДЕТ, А ТЕБЕ ОТОРВУТ РУКИ!!!
import cv2

import sys

from init import gst_stream

sys.path.append('..')

import predictor

cap = cv2.VideoCapture(gst_stream)
predictor.try_detect_frame(cap)
cap.release()
cv2.destroyAllWindows()

print("cv detector was closed")
