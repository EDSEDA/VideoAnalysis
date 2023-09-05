from fastapi import APIRouter, Depends

from typing import List, Union
from api.utils import module_url


r = APIRouter()
BASE = module_url(__name__)

max_limit = 100


@r.get(BASE + '/template', response_model=List[str])
async def get_template_names(supp_id: str, params=Depends(...)):
    """
    В
    """

    return ...
