import pika, sys, os
from threading import Thread
import time
import json

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel, QGridLayout, QLineEdit, QCheckBox
from PyQt5.QtGui import QPixmap

statistics = {
    "Anger": 0,
    "Fear": 0,
    "Happy": 0,
    "Neutral": 0,
    "Sadness": 0,
    "Surprise": 0,
}

healthCheckToCdku = {
    "franchisee": [statistics]
}

class EmulGui(QWidget):

    def __init__(self):
        super().__init__()

        img1 = QPixmap("files/img1.png")
        img1 = img1.scaled(QSize(250, 250))
        pixLbl1 = QLabel()
        pixLbl1.setPixmap(img1)
        lbl1 = QLabel('Ворков Никита')

        lbl111 = QLabel('Anger: ')
        lbl112 = QLabel('20 %')
        lbl121 = QLabel('Fear: ')
        lbl122 = QLabel('30 %')
        lbl131 = QLabel('Happy: ')
        lbl132 = QLabel('25 %')
        lbl141 = QLabel('Neutral: ')
        lbl142 = QLabel('10 %')
        lbl151 = QLabel('Sadness: ')
        lbl152 = QLabel('40 %')
        lbl161 = QLabel('Surprise: ')
        lbl162 = QLabel('12 %')
        # label->setText(text);

        img2 = QPixmap("./files/img2.png")
        img2 = img2.scaled(QSize(250, 250))
        pixLbl2 = QLabel()
        pixLbl2.setPixmap(img2)
        lbl2 = QLabel('Иван Костылев')

        lbl211 = QLabel('Anger: ')
        lbl212 = QLabel('12 %')
        lbl221 = QLabel('Fear: ')
        lbl222 = QLabel('17 %')
        lbl231 = QLabel('Happy: ')
        lbl232 = QLabel('22 %')
        lbl241 = QLabel('Neutral: ')
        lbl242 = QLabel('34 %')
        lbl251 = QLabel('Sadness: ')
        lbl252 = QLabel('45 %')
        lbl261 = QLabel('Surprise: ')
        lbl262 = QLabel('23 %')

        empty = QLabel('')

        grid = QGridLayout()
        grid.setSpacing(5)

        grid.addWidget(pixLbl1, 1, 0, 6, 6)
        grid.addWidget(lbl1, 7, 0, 1, 4)

        grid.addWidget(lbl111, 1, 5, 1, 1)
        grid.addWidget(lbl112, 1, 6, 1, 1)
        grid.addWidget(lbl121, 2, 5, 1, 1)
        grid.addWidget(lbl122, 2, 6, 1, 1)
        grid.addWidget(lbl131, 3, 5, 1, 1)
        grid.addWidget(lbl132, 3, 6, 1, 1)
        grid.addWidget(lbl141, 4, 5, 1, 1)
        grid.addWidget(lbl142, 4, 6, 1, 1)
        grid.addWidget(lbl151, 5, 5, 1, 1)
        grid.addWidget(lbl152, 5, 6, 1, 1)
        grid.addWidget(lbl161, 6, 5, 1, 1)
        grid.addWidget(lbl162, 6, 6, 1, 1)

        grid.addWidget(empty, 8, 0)
        grid.addWidget(empty, 8, 0)

        grid.addWidget(pixLbl2, 9, 0, 6, 6)
        grid.addWidget(lbl2, 15, 0, 1, 4)
        
        grid.addWidget(lbl211, 9, 5, 1, 1)
        grid.addWidget(lbl212, 9, 6, 1, 1)
        grid.addWidget(lbl221, 10, 5, 1, 1)
        grid.addWidget(lbl222, 10, 6, 1, 1)
        grid.addWidget(lbl231, 11, 5, 1, 1)
        grid.addWidget(lbl232, 11, 6, 1, 1)
        grid.addWidget(lbl241, 12, 5, 1, 1)
        grid.addWidget(lbl242, 12, 6, 1, 1)
        grid.addWidget(lbl251, 13, 5, 1, 1)
        grid.addWidget(lbl252, 13, 6, 1, 1)
        grid.addWidget(lbl261, 14, 5, 1, 1)
        grid.addWidget(lbl262, 14, 6, 1, 1)

        self.setLayout(grid)
        self.setGeometry(300, 300, 720, 480)
        self.setWindowTitle('DccEmul')
        self.show()

    def start(self):
        self.connectionReceive = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channelReceive = self.connectionReceive.channel()
        try:
            self.channelReceive.exchange_declare('mediaserver2cdku', exchange_type='direct')
            self.channelReceive.queue_declare(queue='mediaserver2cdku-q1')
        except Exception:
            print('mediaserver2cdku has already declared')
        recvThread = Thread(target=self.handleRegularMessage)
        recvThread.start()

        self.connectionSend = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
        self.channelSend = self.connectionSend.channel()
        try:
            self.channelSend.exchange_declare('cdku2mediaserver', exchange_type='direct')
            self.channelSend.queue_declare(queue='cdku2mediaserver-q1')
        except Exception:
            print('cdku2mediaserver has already declared')
        self.channelSend.queue_bind(queue='cdku2mediaserver-q1', exchange='cdku2mediaserver', routing_key='cdku2mediaserver-q1')
        sendThread = Thread(target=self.sendRegularMessage)
        sendThread.start()

    def regularMessageCallback(self, ch, method, properties, body):

        print("Received:\n" + str(json.loads(body)))

    def handleRegularMessage(self):
        self.channelReceive.basic_consume(queue='mediaserver2cdku-q1', on_message_callback=self.regularMessageCallback, auto_ack=True)
        self.channelReceive.start_consuming()

    def sendAudioSession(self):
        audioSessionToMediaServer["armId"] = int(self.le1b1.text())
        audioSessionToMediaServer["trainId"] = int(self.le2b1.text())
        audioSessionToMediaServer["result"] = self.le3b1.text()
        message = json.dumps(audioSessionToMediaServer)
        props = pika.BasicProperties(headers={'__TypeId__': 'javaKostyl.AudioSessionToMediaServer'})
        self.channelSend.basic_publish(exchange='cdku2mediaserver', routing_key='cdku2mediaserver-q1', body=message, properties=props)
        # print("sendAudioSession")

    def sendAnnounceSession(self):
        announceSessionToMediaServer["armId"] = int(self.le1b2.text())
        announceSessionToMediaServer["trainId"] = int(self.le2b2.text())
        announceSessionToMediaServer["result"] = self.le3b2.text()
        message = json.dumps(announceSessionToMediaServer)
        props = pika.BasicProperties(headers={'__TypeId__': 'javaKostyl.AnnounceSessionToMediaServer'})
        self.channelSend.basic_publish(exchange='cdku2mediaserver', routing_key='cdku2mediaserver-q1', body=message, properties=props)
        # print("sendAnnounceSession")

    def sendChangeOperator(self):
        changeSessionOperatorToMediaServer["oldArmId"] = int(self.le1b3.text())
        changeSessionOperatorToMediaServer["newArmId"] = int(self.le2b3.text())
        changeSessionOperatorToMediaServer["trainId"] = int(self.le3b3.text())
        message = json.dumps(changeSessionOperatorToMediaServer)
        props = pika.BasicProperties(headers={'__TypeId__': 'javaKostyl.ChangeSessionOperatorToMediaServer'})
        self.channelSend.basic_publish(exchange='cdku2mediaserver', routing_key='cdku2mediaserver-q1', body=message, properties=props)
        # print("sendChangeOperator")

    def sendRegularMessage(self):
        while (True):
            if (self.cb.isChecked()):
                healthCheckToMediaServer["connectionState"] = self.le1cb.text()
                message = json.dumps(healthCheckToMediaServer)
                props = pika.BasicProperties(headers= {'__TypeId__': 'javaKostyl.HealthCheckToMediaServer'})
                self.channelSend.basic_publish(exchange='cdku2mediaserver', routing_key='cdku2mediaserver-q1', body=message, properties=props)
                # print("Regular message sent:\n" + message)
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
