from fastapi import APIRouter, Depends, HTTPException
from typing import List, Union

from sqlalchemy import select, insert, delete, text

from api.utils import module_url
from api.context import session
import server.schema as schema
from api.model import Visitor, Emotion, Shop


r = APIRouter()
BASE = module_url(__name__)

max_limit = 100


@r.get(BASE + '/', response_model=List[schema.Shop])
async def get_shops():
    """
    Вернуть все точки работы
    """
    shops = (await session().execute(select(Shop))).scalars()
    return shops


@r.get(BASE + '/{shop_id}', response_model=List[schema.Shop])
async def get_shop(shop_id: int):
    """
    Вернуть магазин
    """
    visitor = (await session().execute(select(Shop).where(Shop.id == shop_id))).scalars().one_or_none()
    if not visitor:
        raise HTTPException(404, "Shop not found")
    return visitor
