# from server.config import settings, server_log
from server.config import settings, server_log
import logging
import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from starlette.responses import RedirectResponse
from api.db import migrate, async_session
from api.context import set_session
from server.routes import emotions
from api.utils import create_routes
from api.rabbit import start_message_consumer


app = FastAPI()


@app.on_event("startup")
async def startup():
    migrate()
    asyncio.create_task(start_message_consumer())
    async with async_session() as session:
        set_session(session)


@app.on_event("shutdown")
async def shutdown():
    pass


create_routes(app, emotions)


def run():
    server_log.info(f"Settings: {settings.model_dump()}")
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)


if __name__ == '__main__':
    run()
