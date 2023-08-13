import logging

from api.config import Settings as BaseSettings

server_log = logging.getLogger(name="server logger")


class Settings(BaseSettings):
    APP_HOST: str = '127.0.0.1'
    APP_PORT: int = 8181
