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

"""
показывать статистику по:
  1 магазину
  2 компании
  3 человеку (возвращаемся к первому вопросу + второму)
"""


@r.get(BASE + '/', response_model=List[schema.Visitor])
async def get_visitors():
    """
    Вернуть всех посетителей
    """
    visitors = (await session().execute(select(Visitor))).scalars()
    return visitors


@r.get(BASE + '/{visitor_id}', response_model=schema.Visitor)
async def get_visitor(visitor_id: int):
    """
    Вернуть посетителя
    """
    visitor = (await session().execute(select(Visitor).where(Visitor.id == visitor_id))).scalars().one_or_none()
    if not visitor:
        raise HTTPException(404, "Visitor not found")
    return visitor
