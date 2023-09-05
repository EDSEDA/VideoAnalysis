from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)


class Settings(BaseModel):
    RM_HOST: str = 'localhost'
    RM_PORT: int = 5672
    RM_USER: str = 'guest'
    RM_PASSWORD: str = 'guest'

    DB_HOST: str = 'localhost'
    DB_PORT: int = 5434
    DB_DATABASE: str = 'postgres'
    DB_USER: str = 'postgres'
    DB_PASSWORD: str = 'mysecretpassword'
    DB_ECHO: bool = False

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()

DB_URL = f'{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_DATABASE}'
ASYNC_DB_URL = f'postgresql+asyncpg://{DB_URL}'
SYNC_DB_URL = f'postgresql://{DB_URL}'

WORK_SCHEMA = 'test_name'

EMOTION_LABELS = ['anger', 'fear', 'happy', 'neutral', 'sadness', 'surprized']


class Paths(BaseModel):
    FACE_CLASSIFIER_PATH: str = "../cfg/face_detector.xml"
    PREDICTION_MODEL_PATH: str = "../learning/data/model.h5"
    CONFIG_PATH: str = "../cfg/emotion_detector.yaml"


paths = Paths()
