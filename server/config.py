from pydantic import BaseModel
import logging

# Настраиваем базовую конфигурацию для логирования
logging.basicConfig(level=logging.INFO)

server_log = logging.getLogger(name="server logger")


class Settings(BaseModel):
    APP_HOST: str = '127.0.0.1'
    APP_PORT: int = 8181
