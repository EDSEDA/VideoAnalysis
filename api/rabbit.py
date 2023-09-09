import json

import pika
from api.config import settings, logging, RABBITMQ_URL
from api.model import Shop, User, Emotion
from pydantic import BaseModel
import aio_pika
from api.context import session
from sqlalchemy import select, insert

exchanger_name = 'camera_to_server'
queue_name = 'camera_to_server'
routing_key = queue_name

credentials = pika.PlainCredentials(username=settings.RM_USER, password=settings.RM_PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=settings.RM_HOST, port=settings.RM_PORT, credentials=credentials))
channel = connection.channel()
channel.exchange_declare(exchange=exchanger_name, exchange_type='direct')
channel.queue_declare(queue=queue_name)
channel.queue_bind(queue=queue_name, exchange=exchanger_name, routing_key=routing_key)


def mq_send(msg: str):
    channel.basic_publish(exchange=exchanger_name, routing_key=routing_key, body=msg.encode())


def mq_recv(callback: callable):  # формат колбэк функции: callback(ch, method, properties, body):
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()


class Message(BaseModel):
    worker_id: int
    anger: int
    fear: int
    happy: int
    neutral: int
    sadness: int
    surprized: int
    age_group: int
    sex: int
    consultation_time: int
    date: int
    placement_point: int = 1


async def save_message_to_database(message_body: str):
    # test msg {"anger":0, "fear":11, "happy":0, "neutral":84, "sadness":14, "surprized":0, "worker_id":17, "age_group":32, "sex":0, "consultation_time":110, "date":1694221445}
    if message_body:
        payload = Message.model_validate(json.loads(message_body))
        logging.info(f'message received: {payload}')
        worker = (await session().execute(select(User).where(User.id == payload.worker_id))).scalars().one_or_none()
        if not worker:
            await session().execute(insert(User).values(dict(id=payload.worker_id,
                                                             name='test_worker')))
        shop = (await session().execute(select(Shop).where(Shop.id == payload.placement_point))).scalars().one_or_none()
        if not shop:
            await session().execute(insert(Shop).values(dict(id=payload.placement_point,
                                                             name='test_shop')))


async def consume_messages(loop):
    connection = await aio_pika.connect_robust(
        RABBITMQ_URL, loop=loop
    )

    async with connection:

        channel: aio_pika.abc.AbstractChannel = await connection.channel()

        queue: aio_pika.abc.AbstractQueue = await channel.declare_queue(
            queue_name
        )

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        await save_message_to_database(message.body.decode())
                    except Exception as e:
                        logging.error(e)


async def start_message_consumer(loop):
    logging.info('Start rabbit message consumer')
    await consume_messages(loop=loop)
