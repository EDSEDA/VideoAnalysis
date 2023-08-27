import pika

from api.config import Settings

exchanger_name = 'camera_to_server'
queue_name = 'camera_to_server'
routing_key = queue_name

credentials = pika.PlainCredentials(username=Settings.RM_USER, password=Settings.RM_PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=Settings.RM_HOST, port=Settings.RM_PORT, credentials=credentials))
channel = connection.channel()
channel.exchange_declare(exchange=exchanger_name, exchange_type='direct')
channel.queue_declare(queue=queue_name)
channel.queue_bind(queue=queue_name, exchange=exchanger_name, routing_key=routing_key)


def mq_send(msg: str):
    channel.basic_publish(exchange=exchanger_name, routing_key=routing_key, body=msg.encode())


def mq_recv(callback: callable):
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()
