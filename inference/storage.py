import face_recognition


identified_face_encodings = []
identified_face_names = []

name = "Kostylev Ivan"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person1.2.jpg")
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
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person3.2.jpg")
face_locations = face_recognition.face_locations(learned_person)
learned_person_encoding = face_recognition.face_encodings(learned_person, face_locations)[0]
identified_face_encodings.append(learned_person_encoding)
identified_face_names.append(name)

name = "Sulimenko Nikita"
learned_person = face_recognition.load_image_file("../learning/data/face_recognition_images/person4.2.jpg")
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

NAMES_LABELS = ["Kostylev Ivan", "Vorkov Nikita", "Bondarenko Andrey", "Sulimenko Nikita", "Popov Alexander", "Karatetskaya Maria", "Unknown Person"]
names_dict = dict()
names_dict["Kostylev Ivan"] = 0
names_dict["Vorkov Nikita"] = 1
names_dict["Bondarenko Andrey"] = 2
names_dict["Sulimenko Nikita"] = 3
names_dict["Popov Alexander"] = 4
names_dict["Karatetskaya Maria"] = 5
names_dict["Unknown Person"] = 6

names = ["Unknown Person"] * 10

recomendations = dict()
recomendations[NAMES_LABELS[0]] = ["шоколад", "вода 0.5л",  "яйца",  "сок 1л",  "сухарики",  "мороженое" ]
recomendations[NAMES_LABELS[1]] = ["мороженое", "сок 1л",  "яйца", "леденец",  "сухарики", "чипсы"  ]
recomendations[NAMES_LABELS[2]] = ["яйца", "сок 1л",  "яйца",  "мороженое",  "чипсы",  "мороженое" ]
recomendations[NAMES_LABELS[3]] = ["мандарины", "леденец",  "яйца",  "чипсы",  "сухарики",  "чипсы" ]
recomendations[NAMES_LABELS[4]] = ["леденец", "шоколад",  "яйца",  "сок 1л",  "сухарики",  "мороженое" ]
recomendations[NAMES_LABELS[5]] = ["шоколад", "вода 0.5л",  "яйца",  "сок 1л",  "сухарики",  "мороженое" ]
recomendations[NAMES_LABELS[6]] = ["шоколад", "яйца",  "мороженое",  "сухарики",  "леденец",  "чипсы" ]

carModels = dict()
carModels[NAMES_LABELS[0]] = "Lada Priora"
carModels[NAMES_LABELS[1]] = "Tesly CyperTrack"
carModels[NAMES_LABELS[2]] = "Tayoty Mark2"
carModels[NAMES_LABELS[3]] = "BMW X5"
carModels[NAMES_LABELS[4]] = "Volkswagen Golf"
carModels[NAMES_LABELS[5]] = "without car"
carModels[NAMES_LABELS[6]] = "Porche 911"

gasStation = dict()
gasStation[NAMES_LABELS[0]] = 1
gasStation[NAMES_LABELS[1]] = 2
gasStation[NAMES_LABELS[2]] = 3
gasStation[NAMES_LABELS[3]] = 4
gasStation[NAMES_LABELS[4]] = 5
gasStation[NAMES_LABELS[5]] = 0
gasStation[NAMES_LABELS[6]] = 0

indexes = dict()
indexes[NAMES_LABELS[0]] = 13
indexes[NAMES_LABELS[1]] = 2024
indexes[NAMES_LABELS[2]] = 69
indexes[NAMES_LABELS[3]] = 21
indexes[NAMES_LABELS[4]] = 100
indexes[NAMES_LABELS[5]] = 7
indexes[NAMES_LABELS[6]] = 7

sails = dict()
sails[NAMES_LABELS[0]] = 5
sails[NAMES_LABELS[1]] = 10
sails[NAMES_LABELS[2]] = 25
sails[NAMES_LABELS[3]] = 50
sails[NAMES_LABELS[4]] = 30
sails[NAMES_LABELS[5]] = 15
sails[NAMES_LABELS[6]] = 15

kostyName = dict()
kostyName[NAMES_LABELS[0]] = "Костылев Иван"
kostyName[NAMES_LABELS[1]] = "Ворков Никита"
kostyName[NAMES_LABELS[2]] = "Бондаренко Андрей"
kostyName[NAMES_LABELS[3]] = "Сулименко Никита"
kostyName[NAMES_LABELS[4]] = "Попов Александр"
kostyName[NAMES_LABELS[5]] = "Каратецкая Мария"
kostyName[NAMES_LABELS[6]] = "Новый клиент"

last_val_iterator = dict()
last_val_iterator[NAMES_LABELS[0]] = 0
last_val_iterator[NAMES_LABELS[1]] = 0
last_val_iterator[NAMES_LABELS[2]] = 0
last_val_iterator[NAMES_LABELS[3]] = 0
last_val_iterator[NAMES_LABELS[4]] = 0
last_val_iterator[NAMES_LABELS[5]] = 0
last_val_iterator[NAMES_LABELS[6]] = 0
last_val_iterator["general"] = False

full_flag = dict()
full_flag[NAMES_LABELS[0]] = False
full_flag[NAMES_LABELS[1]] = False
full_flag[NAMES_LABELS[2]] = False
full_flag[NAMES_LABELS[3]] = False
full_flag[NAMES_LABELS[4]] = False
full_flag[NAMES_LABELS[5]] = False
full_flag[NAMES_LABELS[6]] = False
full_flag["general"] = False

age_last_values = dict()
age_last_values[NAMES_LABELS[0]] = [0] * 50
age_last_values[NAMES_LABELS[1]] = [0] * 50
age_last_values[NAMES_LABELS[2]] = [0] * 50
age_last_values[NAMES_LABELS[3]] = [0] * 50
age_last_values[NAMES_LABELS[4]] = [0] * 50
age_last_values[NAMES_LABELS[5]] = [0] * 50
age_last_values[NAMES_LABELS[6]] = [0] * 50

sex_last_values = dict()
sex_last_values[NAMES_LABELS[0]] = [0] * 50
sex_last_values[NAMES_LABELS[1]] = [0] * 50
sex_last_values[NAMES_LABELS[2]] = [0] * 50
sex_last_values[NAMES_LABELS[3]] = [0] * 50
sex_last_values[NAMES_LABELS[4]] = [0] * 50
sex_last_values[NAMES_LABELS[5]] = [0] * 50
sex_last_values[NAMES_LABELS[6]] = [0] * 50

race_last_values = dict()
race_last_values[NAMES_LABELS[0]] = [0] * 50
race_last_values[NAMES_LABELS[1]] = [0] * 50
race_last_values[NAMES_LABELS[2]] = [0] * 50
race_last_values[NAMES_LABELS[3]] = [0] * 50
race_last_values[NAMES_LABELS[4]] = [0] * 50
race_last_values[NAMES_LABELS[5]] = [0] * 50
race_last_values[NAMES_LABELS[6]] = [0] * 50

emotion_last_values = dict()
emotion_last_values[NAMES_LABELS[0]] = [0] * 50
emotion_last_values[NAMES_LABELS[1]] = [0] * 50
emotion_last_values[NAMES_LABELS[2]] = [0] * 50
emotion_last_values[NAMES_LABELS[3]] = [0] * 50
emotion_last_values[NAMES_LABELS[4]] = [0] * 50
emotion_last_values[NAMES_LABELS[5]] = [0] * 50
emotion_last_values[NAMES_LABELS[6]] = [0] * 50


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []