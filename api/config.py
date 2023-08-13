from pydantic import BaseModel
import logging


logging.basicConfig(level=logging.INFO)


class Settings(BaseModel):
    RM_HOST: str = 'localhost'
    RM_PORT: int = 5672
    RM_USER: str = 'rmuser'
    RM_PASSWORD: str = 'rmpassword'
