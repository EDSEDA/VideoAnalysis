import json

from confluent_kafka import Producer

class Connection:
    def init(self, addr, topic):
        self.__addr = addr
        self.__topic = topic

        producer_config = {
            'bootstrap.servers': self.__addr,
            'client.id': 'python-producer'
        }
        self.producer = Producer(**producer_config)

    def send(self, data: dict):
        self.producer.produce(self.__topic, value=json.dumps(data))
        self.producer.flush()

connection = Connection()