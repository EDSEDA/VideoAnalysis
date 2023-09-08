import pika
from api.config import settings, logging, RABBITMQ_URL
from pydantic import BaseModel
import asyncio
import aiormq

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
    body: str


async def save_message_to_database(message_body: str):
    if message_body:
        logging.info(f'message received: {message_body}')


async def consume_messages():
    connection = await aiormq.connect(url=RABBITMQ_URL)
    channel = await connection.channel()
    await channel.queue_declare(queue_name)
    while True:
        message = await channel.basic_get(queue_name)
        if message:
            message_body = message.body.decode()
            await save_message_to_database(message_body)
        await asyncio.sleep(settings.CHECK_RABBIT_PERIOD)


async def start_message_consumer():
    logging.info('Start rabbit message consumer')
    while True:
        await consume_messages()
