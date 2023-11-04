from typing import Union

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
    visitor_id: int
    datetime_start: Union[datetime, None]
    consultation_time: Union[datetime, None]
    sex: Union[bool, None]
    placement_point: int


class Shop(BaseModel):
    id: int
    name: str
    address: str


class Visitor(BaseModel):
    id: int
    name: str
    lastname: Union[str, None]
    role: Union[str, None]
