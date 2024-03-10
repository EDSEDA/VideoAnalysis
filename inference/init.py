from ultralytics import YOLO
import torch
from torchvision import transforms

import models

GENDER_LABELS = ['male', 'female']
RACES_LABELS = ['white', 'black', 'asian', 'indian', 'others']
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise"]
EMOTION_LABELS_BIN = ['negative', 'neutral', 'positive']

# Переменные для реализации идентфикации
last_identified_person_id = 0
identified_face_embeddings = []

# Переменные для фильтрации идентификации людей
identified_person_frames_counter = 0
identified_required_person_frames = 10
identified_enought_person_frames = False
identified_person_ids = [0] * identified_required_person_frames

# Переменные для фильтрации оценки внешнего вида людей
analyzed_person_frames_counter = 0
analyzed_required_person_frames= 50
analyzed_enought_person_frames = False
age_last_values = [0] * analyzed_required_person_frames
sex_last_values = [0] * analyzed_required_person_frames
rac_last_values = [0] * analyzed_required_person_frames
emo_last_values = [0] * analyzed_required_person_frames

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
model_face.to("cuda")
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