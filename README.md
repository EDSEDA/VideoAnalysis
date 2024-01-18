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
Чтобы заработала библиотека OpenCV c gstreamer бэкэндом, необходимо руками собрать эту библиотеку:
Для python:         
* `git submodule update --init --recursive --depth 1`     
* `cd opencv-python`      
* `git checkout origin/master`        
* `git submodule update --recursive`      
* `export ENABLE_CONTRIB=1`       
* `export CMAKE_ARGS="-DWITH_GSTREAMER=ON"`       
* `python3 -m pip wheel . --verbose`      
* `python3 -m pip install opencv_*.whl`       

### Description

![Work scheme](./doc/work_scheme.png)