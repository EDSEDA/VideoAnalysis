# EDA (emotion decetion app)

### Services:
- [Learning](./learning)
- [Emotion detector:](./inference)
- [Frontend](./client1)
- [Backend](./server)

### Start
- RESTApi server: `run app.py`

Чтобы сразу имитировать rtsp поток вместо использования /dev/video0:
`sudo docker run --device=/dev/video0 --network="host" -it mpromonet/v4l2rtspserver:v0.3.8 -I 127.0.0.1 -P 18554 -u test`

### Description

![Work scheme](./doc/work_scheme.png)