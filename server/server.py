from config import settings, server_log
import logging
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from starlette.responses import RedirectResponse
from api.db import migrate

app = FastAPI()


@app.on_event("startup")
async def startup():
    migrate()


@app.on_event("shutdown")
async def shutdown():
    pass


if __name__ == '__main__':
    server_log.info(f"Settings: {settings.model_dump()}")
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)

