from fastapi import APIRouter, Depends
from typing import List, Union

from sqlalchemy import select, insert, delete, text

from api.utils import module_url
from api.context import session
import server.schema as schema
from api.model import User, Emotion, Shop


r = APIRouter()
BASE = module_url(__name__)

max_limit = 100


@r.get(BASE + '/', response_model=List[schema.User])
async def get_emotions():
    """
    Вернуть всех пользователей
    """
    workers = (await session().execute(select(User))).scalars()
    return workers