import pika
from enum import Enum

from api.config import Settings


class QueueName(Enum):
    test = 'test'
    another = 'another'


credentials = pika.PlainCredentials(username=Settings.RM_USER, password=Settings.RM_PASSWORD)
connection = pika.BlockingConnection(pika.ConnectionParameters(host=Settings.RM_HOST, port=Settings.RM_PORT,
                                                               credentials=credentials))

test_channel = connection.channel()
test_channel.queue_declare(queue='test')

another_channel = connection.channel()
another_channel.queue_declare(queue='another')

queue = dict(
    test=test_channel,
    another=another_channel
)

routing_key = 'test_key'


def mq_send(msg: str, queue_name: QueueName):
    queue[queue_name].basic_publish(exchange='', routing_key=routing_key, body=msg.encode())


# connection.close()
