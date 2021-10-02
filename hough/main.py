#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, time

# 영상 사이즈는 가로세로 640 x 480
Width = 640
Height = 480
# ROI 영역 : 세로 420 ~ 460 만큼 잘라서 사용
Offset = 330
Gap = 40

DELTA_T = 0.05

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


# draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
    return img

# draw rectangle

def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2
    cv2.rectangle(img, (lpos - 5, 15 + offset), (lpos + 5, 25 + offset), (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset), (rpos + 5, 25 + offset), (0, 255, 0), 2)
    cv2.rectangle(img, (center - 5, 15 + offset), (center + 5, 25 + offset), (0, 255, 0), 2)
    cv2.rectangle(img, (330, 15 + offset), (340, 25 + offset), (0, 0, 255), 2)
    return img


def divide_left_right(lines):
    global Width
    # 기울기 절대값이 0 ~ 10 인것만 추출
    low_slope_threshold = 0
    high_slope_threshold = 10

    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)
        if(abs(slope) > low_slope_threshold) and (abs(slope) < high_slope_threshold):
            slopes.append(slope)
            new_lines.append(line[0])

        
    # divide lines left to right

    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]
        x1, y1, x2, y2 = Line
        # 화면에 왼쪽/오른쪽에 있는 선분 중에서 기울기가 음수 / 양수 인것들만 모음
        if(slope < 0) and (x2 < Width/2 - 90):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width/2 + 90):
            right_lines.append([Line.tolist()])
    return left_lines, right_lines

# 기울기와 y절편의 평균값 구하기

def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)

    if size == 0:
        return 0, 0
    
    for line in lines :
        x1, y1, x2, y2 = line[0]
        
        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b


def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap
    m, b = get_line_params(lines)

    if m == 0 and b == 0:
        if left: 
            pos = 0
        if right:
            pos = Width
    else:
        # y 값을 ROI의 세로 중간값으로 지정하여 대입
        y = Gap / 2
        pos = (y - b) / m

        # y 값을 맨 끝 값들로 정해줬을 때의 x값 구함
        b += Offset
        x1 = (Height - b) /float(m)
        x2 = ((Height/2) - b) / float(m)

        cv2.line(img, (int(x1), Height), (int(x2), (Height/2)), (255,0,0), 3)
    return img, pos


def process_image(frame):
    global Width
    global Offset, Gap

    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


    # canny edge
    low_threshold = 60
    high_threshold = 70
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    # HoughLinesP
    roi = edge_img[Offset : Offset+Gap, 0 : Width]
    all_lines = cv2.HoughLinesP(roi, 1, math.pi/180, 30,30,10)

    # divide left, right lines
    if all_lines is None:
        return (0, 640), frame
    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines

    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True)
    lpos = int(lpos)
    rpos = int(rpos)
    # draw lines

    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)

    # draw rectangle
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)

    return (lpos, rpos), frame

def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    arrow_pic = cv2.imread('steer_arrow.png', cv2.IMREAD_COLOR)

    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.76
    arrow_Height = Height/2
    arrow_Width = (arrow_Height * 462)/728

    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (steer_angle) * 2.5, 0.7)    
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width/2 - arrow_Width/2) : (Width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

    cv2.imshow('steer', image)

def start():

    global image, Width, Height

    angle_list = []
        
    steering = PID(0.25, 0, 0.03)

    cap = cv2.VideoCapture('./deepLane/data/kmu_track.mkv')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    fourcc = cv2.VideoWriter_fourcc('D','I','V','X')  


    out = cv2.VideoWriter('./Hough.avi', fourcc, 20, (640, 480))

    while not rospy.is_shutdown():

        ret, image = cap.read()
        cv2.imshow("image", image)

        pos, frame = process_image(image)
        #print('1111111111111111111111111', pos)
        center = (pos[0] + pos[1]) / 2        


        angle = 335 - center
        angle = steering.getPID(angle)

        angle_list.append(angle)


        if(len(angle_list) > 10):
            #print('-----------------')
            avg_angle = 0.0
            for i in range(len(angle_list) - 1, len(angle_list) - 11, -1):
                #print(1)
                avg_angle += angle_list[i]
    
    	    avg_angle = avg_angle / 10
    	    angle = avg_angle
            #print('-----------------')


        if angle > 20:
            angle = 20
        elif angle < -20:
            angle = -20


        draw_steer(frame, angle)
        out.write(frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    start()
    
