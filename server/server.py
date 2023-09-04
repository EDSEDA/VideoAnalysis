import socket
from config import Settings, server_log
import logging
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse


# NL = open("logs/NikitaStatistics.log", "w+")
# PL = open("logs/PetrStatistics.log", "w+")

clientNames = {8282: "Nikita", 8383: "Petr"}
clients = dict.fromkeys(clientNames.keys())


class Client:

    def __init__(self, address, port):
        self.emotions = {'Anger': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sadness': 0, 'Surprized': 0}
        self.address = address
        self.port = port

    def updateEmotionStatistics(self, emotion):
        self.emotions[emotion] = self.emotions[emotion] + 1

    def getEmotionStatistics(self):
        emotionsSum = sum(self.emotions.values())
        emotionsPercentage = dict(zip(self.emotions.keys(), [float("{0:.3f}".format(float(v) / emotionsSum)) for k, v in
                                                             self.emotions.items()]))
        return emotionsPercentage  # тут надо бы еще слить с ключами мапы


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 8181))


def server_run():
    while True:
        data, ip = sock.recvfrom(1024)
        address = ip[0]
        port = int(ip[1])
        emotion = data.decode('utf-8')

        # print (emotion, "received from: ", clientNames[port])

        if clients[port] == None:
            clients[port] = Client(address, port)

        clients[port].updateEmotionStatistics(emotion)

        if clientNames[port] == "Nikita":
            NL.write("Nikita emotion statistics: " + str(clients[port].getEmotionStatistics()) + "\n")
            NL.flush()

        if clientNames[port] == "Petr":
            PL.write("Petr emotion statistics: " + str(clients[port].getEmotionStatistics()) + "\n")
            PL.flush()


if __name__ == '__main__':
    server_log.info(f"Settings: {Settings().model_dump()}")
    server_run()
