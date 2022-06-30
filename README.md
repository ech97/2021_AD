
## 개요
제 5회 국민대 자율주행차 경진대회 프로젝트

## 구현방법
> 1. 허프변환을 이용한 저성능 임베디드보드를 위한 차선인식 자율주행 알고리즘
> 2. Nvidia의 LaneNet을 이용한 차선 Binary Segmentation 및 scikit의 dbscan을 이용한 차선 분류 알고리즘


## Sample Video
> 허프변환을 사용했을 때

![Hough](./Hough.gif)

> Lanenet을 이용하였을떄

![deepLane](./deepLane.gif)

## Usage
```text
0. 'cd ~/catkin_ws/src && roscreate-pkg lanenet rospy std_msgs sensor_msgs cv_bridge'

1. Unzip the File and overwrite to '~/catkin_ws/src'

2. setting excute permission 'sudo chmod +x ./deepLane/src/test_lanenet.py'

3. catkin_make

**4. 'cd ~/catkin_ws/src/deepLane'
**path 설정 문제로 ~/catkin_ws/src가 아닌 ~/catkin_ws/src/deepLane에서 7번을 수행해야합니다


5. 'pip install -r requirements.txt'
	** Please check your require elements's version

6. 'roscore&'

7. 'rosrun deepLane test_lanenet.py' OR 'python src/test_lanenet.py'
```
