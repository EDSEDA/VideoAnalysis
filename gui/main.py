import pika, sys, os
from threading import Thread
import time
import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel, QGridLayout, QLineEdit, QCheckBox

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


        img = QPixmap("./files/img.png")
        pixLbl = QLabel()
        pixLbl.setPixmap(img)
        lbl1 = QLabel('Ворков\nНикита')

        lbl111 = QLabel('Anger: ')
        lbl111 = QLabel('unknown')
        lbl121 = QLabel('Fear: ')
        lbl121 = QLabel('unknown')
        lbl131 = QLabel('Happy: ')
        lbl131 = QLabel('unknown')
        lbl141 = QLabel('Neutral: ')
        lbl141 = QLabel('unknown')
        lbl151 = QLabel('Sadness: ')
        lbl151 = QLabel('unknown')
        lbl161 = QLabel('Surprise: ')
        lbl161 = QLabel('unknown')
        label->setText(text);

        img1 = QPixmap("./files/img_1.png")
        pixLbl1 = QLabel()
        pixLbl1.setPixmap(img1)
        lbl2 = QLabel('Иван\nКостылев')


#############################################

        grid.addWidget(pixLbl, 1, 0, 4, 4)
        grid.addWidget(lbl1, 3, 0, 2, 4)

        grid.addWidget(lbl111, 1, 3, 3, 1)
        grid.addWidget(lbl111, 1, 3, 3, 1)
        grid.addWidget(lbl121, 1, 3, 3, 1)
        grid.addWidget(lbl121, 1, 3, 3, 1)
        grid.addWidget(lbl131, 1, 3, 3, 1)
        grid.addWidget(lbl131, 1, 3, 3, 1)
        grid.addWidget(lbl141, 1, 3, 3, 1)
        grid.addWidget(lbl141, 1, 3, 3, 1)
        grid.addWidget(lbl151, 1, 3, 3, 1)
        grid.addWidget(lbl151, 1, 3, 3, 1)
        grid.addWidget(lbl161, 1, 3, 3, 1)
        grid.addWidget(lbl161, 1, 3, 3, 1)

        grid.addWidget(pixLbl1, 4, 0, 4, 4)
        grid.addWidget(lbl2, 6, 0, 2, 4)

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
