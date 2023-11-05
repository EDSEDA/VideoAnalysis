from typing import Union, List, Optional

from pydantic import BaseModel
from datetime import date


class Emotions(BaseModel):
    anger: int
    fear: int
    happy: int
    neutral: int
    sadness: int
    surprized: int


class VisitorEmotions(Emotions):
    visitor_id: int
    datetime_start: Union[date, None]
    consultation_time: Union[date, None]
    sex: Union[bool, None]
    placement_point: int


class WithId(BaseModel):
    id: int


class Shop(BaseModel):
    name: str
    address: str


class ShopData(WithId):
    name: str
    address: str


class Visitor(BaseModel):
    id: int
    name: str
    lastname: Union[str, None]
    role: Union[str, None]
    age: Union[int, None]
    sex: Union[bool, None]


class ShopInfo(Shop):
    visitors: List[VisitorEmotions]


class ShortInfoQueryParams(BaseModel):
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    sex: Optional[bool] = None
    age: Optional[int] = None
    shop_id: Optional[int] = None
