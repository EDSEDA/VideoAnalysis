from ultralytics import YOLO
import torch
from torchvision import transforms
import yaml

import grifon.mqbroker.kafka_client as kafka
import grifon.config as cfg
import models

GENDER_LABELS = ['male', 'female']
RACES_LABELS = ['white', 'black', 'asian', 'indian', 'others']
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise"]
EMOTION_LABELS_BIN = ['negative', 'neutral', 'positive']
CONFIG_PATH: str = "cfg/emotion_detector.yaml"

# Чтение конфига
with open(CONFIG_PATH, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

gst_stream = config["gst_stream"]
kafka_addr = config["kafka_addr"]
kafka_topic = config["kafka_topic"]
cash_register_id = config["cash_register_id"]

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
model_person = YOLO('models/video/yolov8n.pt')
model_person.to("cuda")
print("Person model initialized")

model_face = YOLO('models/video/yolov8n-face.pt')
model_face.to("cuda")
print("Face model initialized")

# tracker = DeepSort(max_age=50) ## ToDo начала падать. нужно разобраться почему
print("DeepSort tracker have initialized")

model_age = models.AgeEstimatorModel()
model_age.load_state_dict(torch.load("models/video/age_model_weights.pth", map_location=device))
model_age.eval()
print("Age model have initialized")

model_gen = models.SexEstimatorModel()
model_gen.load_state_dict(torch.load("models/video/sex_model_weights.pth", map_location=device))
model_gen.eval()
print("Gender model have initialized")

model_rac = models.RaceEstimatorModel(5)
model_rac.load_state_dict(torch.load("models/video/race_model_weights.pth", map_location=device))
model_rac.eval()
print("Race model have initialized")

model_emo = models.EmotionBinEstimatorModel(3)
model_emo.load_state_dict(torch.load("models/video/emotion_bin_model_weights.pth", map_location=device))
model_emo.eval()
print("Emotion model have initialized")

kafka_client = kafka.KafkaClient(kafka_addr)