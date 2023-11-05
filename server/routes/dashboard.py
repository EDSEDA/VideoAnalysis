from fastapi import APIRouter, Depends, HTTPException
from typing import List, Union

from sqlalchemy import select, insert, delete, text

from api.utils import module_url
from api.context import session
import server.schema as schema
from api.model import Visitor, Emotion, Shop
from api.config import WORK_SCHEMA


r = APIRouter()
BASE = module_url(__name__)

max_limit = 100


@r.get(BASE + '/shops', response_model=List[schema.ShopInfo])
async def get_shops_info(params=Depends(schema.ShortInfoQueryParams)):
    """
    Вернуть информацию по всем магазинам и клиентам
    """
    q = f"""
Select * from {WORK_SCHEMA}.{Shop.__tablename__} s
left join {WORK_SCHEMA}.{Emotion.__tablename__} e on s.id = e.placement_point
left join {WORK_SCHEMA}.{Visitor.__tablename__} v on e.visitor_id = v.id
"""

    where, args = [], dict()
    if params.date_from:
        where.append(f'e.datetime_start>:date_from')
        args['date_from'] = params.date_from
    if params.date_to:
        where.append(f"(datetime_start + (consultation_time * interval '1 second'))<:date_to")
        args['date_to'] = params.date_to
    if params.sex:
        where.append(f'v.sex = :sex')
        args['sex'] = bool(params.sex)
    if params.age:
        where.append(f'v.age = :age')
        args['age'] = params.age
    if params.shop_id:
        where.append(f's.id = :shop_id')
        args['shop_id'] = params.shop_id

    if where:
        q += 'where ' + ' and '.join(where)

    shops = (await session().execute(text(q), args)).scalars()
    return shops

