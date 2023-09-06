from fastapi import APIRouter, Depends

from typing import List, Union
from api.utils import module_url
from server.schema import UserEmotions


r = APIRouter()
BASE = module_url(__name__)

max_limit = 100


@r.get(BASE + '/', response_model=List[UserEmotions])
async def get_emotions():
    """
    Вернуть все значения эмоций всех пользователей
    """

    return ...
