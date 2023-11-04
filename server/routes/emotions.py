from fastapi import APIRouter, Depends
from typing import List, Union

from sqlalchemy import select, insert, delete, text

from api.utils import module_url
from api.context import session
from server.schema import UserEmotions
from api.model import Visitor, Emotion, Shop


r = APIRouter()
BASE = module_url(__name__)

max_limit = 100


@r.get(BASE + '/', response_model=List[UserEmotions])
async def get_emotions():
    """
    Вернуть все значения эмоций всех пользователей
    """
    emotions = (await session().execute(select(Emotion))).scalars()
    return emotions
