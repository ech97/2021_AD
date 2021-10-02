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



