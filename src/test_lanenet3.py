#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Goal:

기존 Lanenet Network를 통해 수행할 수 있었던 lane Segmentation은 이번 대회의 목적에 부합하지 않는다 생각하여 생략하여 연산량을 줄였습니다.
'''



import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import lanenet
import lanenet_postprocess
import parse_config_utils

CFG = parse_config_utils.lanenet_cfg

DELTA_T = 0.05


#-----------------PID----------------#
#------------------------------------#
class PID:

    def __init__(self, kp, ki, kd):

        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.err_sum = 0.0
        self.delta_err = 0.0
        self.last_err = 0.0

    def getPID(self, error):

        err = error
        self.err_sum += err * DELTA_T
        self.delta_err = err - self.last_err
        self.last_err = err

        self.p = self.kp * err
        self.i = self.ki * self.err_sum
        self.d = self.kd * (self.delta_err * DELTA_T)

        self.u = self.p + self.i + self.d

        return self.u
#------------------------------------#





#-----------------LANENET----------------#
#----------------------------------------#
class lanenet_detector():

    def __init__(self):
        self.weights_path = './model/new_model/tusimple_lanenet.ckpt'
        self.image_path = './data/kmu_track.mkv'
        self.init_lanenet()

        #return parser.parse_args()

    def init_lanenet(self):
        
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

        net = lanenet.LaneNet(phase='test', cfg=CFG)
        self.binary_seg_ret= net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)

        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            self.variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            self.variables_to_restore = self.variable_averages.variables_to_restore()

        # define saver
        saver = tf.train.Saver(self.variables_to_restore)
        saver.restore(sess=self.sess, save_path=self.weights_path)



    def test_lanenet(self, image):
        
        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 0.45


        loop_times = 1
        for i in range(loop_times):
            binary_seg_image = self.sess.run(
                [self.binary_seg_ret],
                feed_dict={self.input_tensor: [image]}
            )
        
        binary_seg_image[0] = np.squeeze(binary_seg_image[0])
        
        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            source_image=image_vis
        )



        mask_image = postprocess_result['mask_image']
        mask_image = cv2.resize(mask_image, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.cvtColor(np.uint8(mask_image), cv2.COLOR_BGR2RGB)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        ret, mask_image = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
        
        return mask_image
#----------------------------------------#



#----------------POST PROCESS------------#
#----------------------------------------#
class _detect_line:
    def __init__(self):

        #----Img Variable----#
        self.width = 640
        self.height = 480
        self.roi_lanenet = 300
        
        self.roi_process= 290
        self.roi_car = 20
        self.roi_correct = self.roi_process - self.roi_car


        #self.dst = np.float32([(0, 0), (640, 0), (0, 290), (640, 290)])
        self.dst = np.float32([(0, 0), (640, 0), (640, 480), (0, 480)])
        self.src = np.float32([(200, 480 - 200), (640 - 125, 480 - 200), (640, 480 - 125), (10, 480 - 100)])


        #----DBSCAN----#
        self.eps = 0.17

        self.min_samples=20

        self.lane_coords = []

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

        # 슬라이딩 윈도우 색상
        self.window_color = (224, 145, 145)

        self.angle = [0, 0, 0]
        self.curvature = [0, 0, 0]



    #----------POLYFIT----------#
    #---------------------------#
    def line_detect(self, mask):
	
	
        num_lane_coords = len(self.lane_coords)
        
        print('---------')

        count = 0

        for i in range(3):
            self.curvature[i] = 0

        for lane_index in range(num_lane_coords):

            # Maximum lane detect
            if(count >= 3):
                break

            pixel = self.lane_coords[lane_index].shape[0]
            
            # Print(Num of Lane, and Pixels)
            print(lane_index, len(self.lane_coords[lane_index]))
    

            nonzero_y = self.lane_coords[lane_index][:, 1]
            nonzero_x = self.lane_coords[lane_index][:, 0]

            if(len(nonzero_y) > 0 and len(nonzero_x) > 0):
                fit_param = np.polyfit(nonzero_y, nonzero_x, 2)

            # When no pixel is detected (Exception Handling)
            else:
                fit_param = [0, 0, 0]
                
            
            #계수는 첫번째꺼(가장 많은 pixel)를 기준으로한걸 사용
            plot_y = np.linspace(0, mask.shape[0] - 61, mask.shape[0] - 60)

            plot_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]

    
            # Lane Drawing을 위해 int32로 캐스팅 및 x범위 클리핑
            plot_y = np.array(plot_y, np.int32)
            plot_x = np.clip(np.array(plot_x, np.int32), 0, 319)

            # Polyfit Drawing        
            mask[plot_y, plot_x] = (155, 45, 175)

            self.angle[count] = fit_param[0]


            y_eval = np.max(plot_y)
            self.curvature[count] =  ((1 + (2*fit_param[0] *y_eval + fit_param[1])**2) **1.5) / np.absolute(2*fit_param[0])
            
            count = count + 1


        print(self.angle)
        print(self.curvature)
        
        return mask
        



    #-----------DBSCAN----------#
    #---------------------------#
    def _embedding_feats_dbscan_cluster(self):
              
        db = DBSCAN(self.eps, self.min_samples)
        try:
            features = StandardScaler().fit_transform(self.coord_bird)
            db.fit(features)
        
        except Exception as err:
            #log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret


    def apply_lane_feats_cluster(self, img_bin):
        
	    # 흰색점의 좌표
        idx = np.where(img_bin == 255)
        self.coord_bird = np.vstack((idx[1], idx[0])).transpose()


        mask = np.zeros(shape=[img_bin.shape[0], img_bin.shape[1], 3], dtype=np.uint8)

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster()

        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']

        if db_labels is None:
            return mask

        self.lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((self.coord_bird[idx][:, 1], self.coord_bird[idx][:, 0]))
	
            if(index > 5 or len(pix_coord_idx[0]) > 20000 or len(pix_coord_idx[0]) < 300):
                continue

            mask[pix_coord_idx] = self._color_map[index]
            self.lane_coords.append(self.coord_bird[idx])

        lane_coords_num = len(self.lane_coords)

        # Sort by Num Of Pixels
        if(lane_coords_num > 0):
                for i in range(lane_coords_num - 1):
                    for j in range(lane_coords_num - 1):
                        if(len(self.lane_coords[j+1]) > len(self.lane_coords[j])):
                            t = self.lane_coords[j]
                            self.lane_coords[j] = self.lane_coords[j+1]
                            self.lane_coords[j+1] = t


        return mask
    #---------------------------#



    def draw_steer(self, image, steer_angle):
        width = self.width
        height = self.height

        arrow_pic = cv2.imread('./src/steer_arrow.png', cv2.IMREAD_COLOR)

        origin_height = arrow_pic.shape[0]
        origin_width = arrow_pic.shape[1]
        steer_wheel_center = origin_height * 0.74
        arrow_height = height/2
        arrow_width = (arrow_height * 462)/728

        matrix = cv2.getRotationMatrix2D((origin_width/2, steer_wheel_center), (steer_angle) * 1.5, 0.7)    
        arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_width+60, origin_height))
        arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_width, arrow_height), interpolation=cv2.INTER_AREA)

        gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)


        arrow_roi = image[arrow_height: height, (width/2 - arrow_width/2) : (width/2 + arrow_width/2)]

        #print(arrow_pic.shape, arrow_roi.shape)


        arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
        res = cv2.add(arrow_roi, arrow_pic)
        image[(height - arrow_height): height, (width/2 - arrow_width/2): (width/2 + arrow_width/2)] = res




    def img_process(self, mask_image):
        '''
        0. masking image 받아오기
        1. 이미지 크기 복구
        2. roi 및 birdeye 적용
        3. dbscan 사용
        4. lane detect
        '''
        mask_image = cv2.resize(mask_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        mask = self.apply_lane_feats_cluster(mask_image)

        # polyfit 수행
        result = self.line_detect(mask)
        cv2.imshow("polyfit", result)

        result = cv2.resize(result, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("result", result)

        return result

#----------------------------------------#











if __name__ == '__main__':
    rospy.init_node('lanenet_node')


    Lanenet = lanenet_detector()
    detect_line = _detect_line()
    pid = PID(22, 0, 10)

    cap = cv2.VideoCapture('./data/kmu_track.mkv')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc('D','I','V','X')  

    out = cv2.VideoWriter('./SaveVideo.avi', fourcc, 20, (640, 480))

    while (cap.isOpened()):

        _, frame = cap.read()

        image = frame.copy()

        # Masking Image 추출
        image[:detect_line.roi_lanenet, :] = 0
        mask_image = Lanenet.test_lanenet(image)
        cv2.imshow("pure lanenet", mask_image)
	
	
        result = detect_line.img_process(mask_image)
        result = cv2.add(result, image)

        steer = pid.getPID(detect_line.angle[0] + 0.5 * detect_line.angle[1])
        
        print('steer: ', steer)


        #detect_line.draw_steer(result, steer)

        frame[detect_line.roi_lanenet:, :] = result[detect_line.roi_lanenet:, :]
        cv2.imshow("result", frame)
        #cv2.imshow("original", image)
        out.write(frame)

        if (cv2.waitKey(20)) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
