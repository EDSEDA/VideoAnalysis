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
    user_id: int
    datetime: datetime
