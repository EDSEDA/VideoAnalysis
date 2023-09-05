# from server.config import settings, server_log
from server.config import settings, server_log
import logging
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from starlette.responses import RedirectResponse
from api.db import migrate
from server.routes import emotions
from api.utils import create_routes

app = FastAPI()


@app.on_event("startup")
async def startup():
    migrate()


@app.on_event("shutdown")
async def shutdown():
    pass


create_routes(app, [emotions])


def run():
    server_log.info(f"Settings: {settings.model_dump()}")
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)


if __name__ == '__main__':
    run()
