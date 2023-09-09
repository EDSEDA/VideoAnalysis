import json
import time
from threading import Thread

import os
import pika
import sys
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QLabel

guiToServerRegular = {
    "userName": "Zloy Director"
}

workerDiagnostic = {
    "workerId": 0,
    "anger": 0,
    "fear": 0,
    "happy": 0,
    "neutral": 0,
    "sadness": 0,
    "surprized": 0,
}

serverToGuiResponse = {
    "workers": [workerDiagnostic]
}

exchangerGuiToServer = "GuiToServer"
queueGuiToServer = "GuiToServer-q1"
exchangerServerToGui = "ServerToGui"
queueServerToGui = "ServerToGui-q1"


class EmulGui(QWidget):

    def __init__(self):
        super().__init__()

        img1 = QPixmap("files/img1.png")
        img1 = img1.scaled(QSize(250, 250))
        pixLbl1 = QLabel()
        pixLbl1.setPixmap(img1)
        lbl1 = QLabel('Ворков Никита')

        lbl111 = QLabel('Anger: ')
        self.lbl112 = QLabel('20 %')
        lbl121 = QLabel('Fear: ')
        self.lbl122 = QLabel('30 %')
        lbl131 = QLabel('Happy: ')
        self.lbl132 = QLabel('25 %')
        lbl141 = QLabel('Neutral: ')
        self.lbl142 = QLabel('10 %')
        lbl151 = QLabel('Sadness: ')
        self.lbl152 = QLabel('40 %')
        lbl161 = QLabel('Surprise: ')
        self.lbl162 = QLabel('12 %')
        # label->setText(text);

        img2 = QPixmap("./files/img2.png")
        img2 = img2.scaled(QSize(250, 250))
        pixLbl2 = QLabel()
        pixLbl2.setPixmap(img2)
        lbl2 = QLabel('Иван Костылев')

        lbl211 = QLabel('Anger: ')
        self.lbl212 = QLabel('12 %')
        lbl221 = QLabel('Fear: ')
        self.lbl222 = QLabel('17 %')
        lbl231 = QLabel('Happy: ')
        self.lbl232 = QLabel('22 %')
        lbl241 = QLabel('Neutral: ')
        self.lbl242 = QLabel('34 %')
        lbl251 = QLabel('Sadness: ')
        self.lbl252 = QLabel('45 %')
        lbl261 = QLabel('Surprise: ')
        self.lbl262 = QLabel('23 %')

        empty = QLabel('')

        grid = QGridLayout()
        grid.setSpacing(5)

        grid.addWidget(pixLbl1, 1, 0, 6, 6)
        grid.addWidget(lbl1, 7, 0, 1, 4)

        grid.addWidget(lbl111, 1, 6, 1, 1)
        grid.addWidget(self.lbl112, 1, 7, 1, 1)
        grid.addWidget(lbl121, 2, 6, 1, 1)
        grid.addWidget(self.lbl122, 2, 7, 1, 1)
        grid.addWidget(lbl131, 3, 6, 1, 1)
        grid.addWidget(self.lbl132, 3, 7, 1, 1)
        grid.addWidget(lbl141, 4, 6, 1, 1)
        grid.addWidget(self.lbl142, 4, 7, 1, 1)
        grid.addWidget(lbl151, 5, 6, 1, 1)
        grid.addWidget(self.lbl152, 5, 7, 1, 1)
        grid.addWidget(lbl161, 6, 6, 1, 1)
        grid.addWidget(self.lbl162, 6, 7, 1, 1)

        grid.addWidget(empty, 8, 0)
        grid.addWidget(empty, 8, 0)

        grid.addWidget(pixLbl2, 9, 0, 6, 6)
        grid.addWidget(lbl2, 15, 0, 1, 4)

        grid.addWidget(lbl211, 9, 6, 1, 1)
        grid.addWidget(self.lbl212, 9, 7, 1, 1)
        grid.addWidget(lbl221, 10, 6, 1, 1)
        grid.addWidget(self.lbl222, 10, 7, 1, 1)
        grid.addWidget(lbl231, 11, 6, 1, 1)
        grid.addWidget(self.lbl232, 11, 7, 1, 1)
        grid.addWidget(lbl241, 12, 6, 1, 1)
        grid.addWidget(self.lbl242, 12, 7, 1, 1)
        grid.addWidget(lbl251, 13, 6, 1, 1)
        grid.addWidget(self.lbl252, 13, 7, 1, 1)
        grid.addWidget(lbl261, 14, 6, 1, 1)
        grid.addWidget(self.lbl262, 14, 7, 1, 1)

        self.setLayout(grid)
        self.setGeometry(300, 300, 720, 480)
        self.setWindowTitle('EDA')
        self.show()

    def start(self):
        self.connectionReceive = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channelReceive = self.connectionReceive.channel()
        self.channelReceive.exchange_declare(exchangerServerToGui, exchange_type='direct')
        self.channelReceive.queue_declare(queue=queueServerToGui)

        recvThread = Thread(target=self.handleRegularMessage)
        recvThread.start()

        self.connectionSend = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
        self.channelSend = self.connectionSend.channel()
        self.channelSend.exchange_declare(exchangerGuiToServer, exchange_type='direct')
        self.channelSend.queue_declare(queue=queueGuiToServer)

        self.channelSend.queue_bind(queue=queueGuiToServer, exchange=exchangerGuiToServer, routing_key=queueGuiToServer)
        sendThread = Thread(target=self.sendRegularMessage)
        sendThread.start()

    def regularMessageCallback(self, ch, method, properties, body):
        serverToGuiResponse = json.loads(body)

        for worker in serverToGuiResponse["workers"]:
            if worker["workerId"] == 0:
                self.lbl112.setText(str(worker["anger"]) + "%")
                self.lbl122.setText(str(worker["fear"]) + "%")
                self.lbl132.setText(str(worker["happy"]) + "%")
                self.lbl142.setText(str(worker["neutral"]) + "%")
                self.lbl152.setText(str(worker["sadness"]) + "%")
                self.lbl162.setText(str(worker["surprized"]) + "%")
            elif worker["workerId"] == 1:
                self.lbl212.setText(str(worker["anger"]) + "%")
                self.lbl222.setText(str(worker["fear"]) + "%")
                self.lbl232.setText(str(worker["happy"]) + "%")
                self.lbl242.setText(str(worker["neutral"]) + "%")
                self.lbl252.setText(str(worker["sadness"]) + "%")
                self.lbl262.setText(str(worker["surprized"]) + "%")

        print("Received body:\n" + str(json.loads(body)))

    def handleRegularMessage(self):
        self.channelReceive.basic_consume(queue=queueServerToGui, on_message_callback=self.regularMessageCallback, auto_ack=True)
        self.channelReceive.start_consuming()

    def sendRegularMessage(self):
        while (True):
            # guiToServerRegular["userName"] = "" # todo когда-нибудь чекать кто смотрит статистику
            message = json.dumps(guiToServerRegular)
            self.channelSend.basic_publish(exchange=exchangerGuiToServer, routing_key=queueGuiToServer, body=message)

            time.sleep(5)


def main():
    app = QApplication(sys.argv)
    gui = EmulGui()
    gui.start()
    sys.exit(app.exec_())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
