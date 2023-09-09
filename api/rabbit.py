import json

import pika
from api.config import settings, logging, RABBITMQ_URL
from pydantic import BaseModel
import aio_pika

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
    anger: int
    fear: int
    happy: int
    neutral: int
    sadness: int
    surprized: int
    worker_id: int
    age_group: int
    sex: int
    consultation_time: int
    date: int


async def save_message_to_database(message_body: str):
    if message_body:
        payload = Message.model_validate(json.loads(message_body))
        logging.info(f'message received: {payload}')


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
                    await save_message_to_database(message.body.decode())


async def start_message_consumer(loop):
    logging.info('Start rabbit message consumer')
    await consume_messages(loop=loop)
