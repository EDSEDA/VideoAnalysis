from pydantic import BaseModel
from datetime import datetime


class Emotions(BaseModel):
    anger: int
    fear: int
    happy: int
    neutral: int
    sadness: int
    surprized: int


class UserEmotions(Emotions):
    worker_id: int
    datetime: datetime
    sex: bool
    placement_point: int


class Shop(BaseModel):
    id: int
    name: str


class User(BaseModel):
    id: int
    name: str
    lastname: str
    role: str
