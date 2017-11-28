#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from future_car_capstone.srv import *
# impot roslib.load_manifest future_car_capstone.
from cv_bridge import CvBridge, CvBridgeError

import rospy
import roslib

import pickle
from tempfile import TemporaryFile

import glob
import os
from math import pi

from yolo import MyDarknet
# from carView import calibClassCar,ViewTransform

# flag
flag_1_subscribeImg_2_loadImgFile_3_image2video = 2  ######################
flag_origin_to_homography = 1
flag_homography_to_drawCircle = 1
flag_drawCircle_to_carView = 1

flag_1_use_YOLO_2_not_use_YOLO = 1

flag_1_fisheye_2_non_fisheye = 1
flag_1_measureONLY_2_kalman = 2

flag_saveImg = 1
flag_publishImg = 1

flag_is_compressed_image = 0
flag_go_to_wrapper_or_save_subImage = 1 #################################

# flag which does not need fixing anymore
flag_saveImg_which_detect_checkerboard = 0
flag_load_calibrated_result = 1
flag_load_detected_result = 0
flag_print = 1

# parameter
ROS_TOPIC = 'desktop/image_color'#/compressed'
Qfactor = 0.1
QfactorProcessNoise = 0.01

path_load_image = []
path_save_image = []
path_carView_undistort_image = []
kernelSize_close = (50, 50)
kernelSize_open = (30, 30)

def setParam():
    global path_load_image, path_save_image, kernelSize_close, kernelSize_open
    global flag_origin_to_homography, flag_homography_to_drawCircle, path_carView_undistort_image

    #non_fisheye
    if flag_1_fisheye_2_non_fisheye == 2:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/'
        kernelSize_close = (50, 50)
        kernelSize_open = (30, 30)

        if flag_origin_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/cctvView/*.png'
        elif flag_homography_to_drawCircle == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/cctvView_homography/*.png'

    #fisheye
    elif flag_1_fisheye_2_non_fisheye == 1:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/'
        path_carView_undistort_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView_undistort/'
        kernelSize_close = (50, 50)
        kernelSize_open = (30, 30)

        if flag_origin_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView/*.png'
        elif flag_homography_to_drawCircle == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView_homography/*.png'

    else:
        print('wrong flag_1_fisheye_2_non_fisheye : ', flag_1_fisheye_2_non_fisheye)

nameOf_pickle_Checkerboard_Detection = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/detect_result_jaesung_171021_1600_delete_files.pickle'
path_pickle_calibration_variable = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/calib_result_JS_fisheye.pickle'
boxColor = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0))
fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0
shape_imgHomography=(616, 565) # (580, 481) #(616, 565)


class CalibClass:
    flag_first_didHomography = 1

    def startHomography(self, frame_JS):

        if self.flag_first_didHomography == 1:
            # cctvView to ground
            srcPixel_Img = np.array([
                [145, 117], [193, 113], [245, 108], [418, 104],
                [119, 190], [169, 187], [224, 185], [418, 185],

                [74, 342], [124, 350],
                [57, 414]

                # # 1288 * 964, 171122 data
                # [145, 117], [653, 116], [748, 93],
                # [539, 271], [622, 257], [719, 238],
                #
                # [511, 543], [650, 549]
            ])

            dstPixel_ground = np.array([
                [55, 0], [145, 0], [235, 0], [508, 0],
                [55, 155], [145, 155], [235, 155], [508, 155],

                [55, 405], [145, 405],
                [55, 510]

                #1288 * 964, 171122 data
                # [55, 0], [145, 0], [235, 0],
                # [55, 155], [145, 155], [235, 155],
                # [145, 405], [235, 405]
            ])


            # for i in range(dstPixel_ground.shape[0]):
            #     dstPixel_ground[i][0] = - 55 + dstPixel_ground[i][0]


            self.homography_RANSAC, mask = cv2.findHomography(srcPoints=srcPixel_Img,
                                                              dstPoints=dstPixel_ground,
                                                              method=0)
                                                              #method=cv2.RANSAC)
                                                              # method=cv2.RANSAC, ransacReprojThreshold=10)
                                                              # method=cv2.RHO)
            print('homography is ', self.homography_RANSAC)

            # do not calculate homography again
            self.flag_first_didHomography = 0

        global shape_imgHomography
        frame_homography_RANSAC = cv2.warpPerspective(frame_JS, self.homography_RANSAC, shape_imgHomography)

        # cv2.namedWindow('Transformation of undistorted frame of JS fisheye camera using homography')
        # cv2.imshow('Transformation of undistorted frame of JS fisheye camera using homography', frame_homography_RANSAC)
        # cv2.waitKey(1)

        return frame_homography_RANSAC  # frame_homography_JS

class KalmanFilterClass:
    def __init__(self):
        # self.processNoise = np.zeros((2,2), dtype=np.float32)
        global QfactorProcessNoise
        self.processNoise = np.eye(2, dtype=np.float32) * QfactorProcessNoise

        global Qfactor
        self.measurementNoise = np.eye(2, dtype=np.float32) * Qfactor
        self.stateTransitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)

    def initialMeasurement(self, z):
        # print('initial measurement')
        self.x_post = np.array([z, 0], dtype=np.float32).reshape((2,1)) #[width, d(width)] or [height, d(height)]
        self.p_post = np.eye(2, dtype=np.float32)
        # print('x_post and p_post is')
        # print(self.x_post)
        # print(self.p_post)

        self.x_pri = np.array((z, 0), dtype=np.float32).reshape((2,1))
        self.p_pri = np.eye(2, dtype=np.float32)
        # print('x_pri and p_pri is')
        # print(self.x_pri)
        # print(self.p_pri)

        # this value is to save the prior position(x_k-1) to get the current velocity(v_k)
        self.z_prior = z


    def timeUpdate(self):
        # print('time update')
        self.x_pri = np.matmul(self.stateTransitionMatrix, self.x_post)
        self.p_pri = np.matmul(np.matmul(self.stateTransitionMatrix, self.p_post), self.stateTransitionMatrix.transpose()) + self.processNoise
        # print('x_pri and p_pri is')
        # print(self.x_pri)
        # print(self.p_pri)

    def measurementUpdate(self, z):
        # print('measurement update')

        z_2by1 = np.array((z, z-self.z_prior), dtype=np.float32).reshape((2,1)) #[position, velocity]

        # this value is to save the prior position(x_k-1) to get the current velocity(v_k)
        self.z_prior = z

        self.kalmanGain = np.matmul(self.p_pri, np.linalg.inv(self.p_pri + self.measurementNoise))
        self.x_post = self.x_pri + np.matmul(self.kalmanGain, z_2by1 - self.x_pri) #[width, d(width)] or [height, d(height)]
        self.p_post = (np.eye(2, dtype=np.float32) - self.kalmanGain) * self.p_pri

        # print('x_post and p_post is')
        # print(self.x_post)
        # print(self.p_post)

        self.timeUpdate()

class LineSegClass:
    flag_imshow_on = 0
    flag_print_on = 0
    flag_first_init_kalman = 1
    end_of_arrow = None
    theta = 0

    def __init__(self, flag_imshow_on, flag_print_on):
        self.flag_imshow_on = flag_imshow_on
        self.flag_print_on = flag_print_on

        global flag_1_measureONLY_2_kalman

    def whiteLineSegmentation(self, frame):

        # print('shape of the frame is ', np.shape(frame))
        frame = cv2.resize(frame, (1062, 598))

        if self.flag_imshow_on == 1:
            cv2.namedWindow('origin_figure')
            cv2.imshow('origin_figure', frame)
            cv2.waitKey(0)

        #convert rgb to hsv
        img_hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]

        if self.flag_imshow_on == 1:
            cv2.namedWindow('hue figure')
            cv2.imshow('hue figure', img_hue)
            cv2.waitKey(0)

        #start to opening the image(morpoholgy)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        #np.zeros((300, img_hue.shape[1]), dtype=np.uint8)
        img_open = cv2.morphologyEx(img_hue, cv2.MORPH_OPEN, kernel)
        print('dtype check', img_open.dtype)
        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_open')
            cv2.imshow('img_open', img_open)
            print('max of img_open is ', np.max(img_open))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        img_sub = np.zeros(img_hue.shape, dtype=np.uint8)
        for i in range(0, img_hue.shape[0]):
            for j in range(0, img_hue.shape[1]):
                if img_hue[i, j] < img_open[i, j]:
                    img_sub[i, j] = 0
                else:
                    img_sub[i, j] = img_hue[i, j] - img_open[i, j]

        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_sub')
            cv2.imshow('img_sub', img_sub)
            cv2.waitKey(0)
        if self.flag_print_on == 2:
            print('shape of the img_sub is ', img_sub.shape)

        # start to closing the image(morpoholgy)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        img_close = cv2.morphologyEx(img_sub, cv2.MORPH_OPEN, kernel_close)
        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_close')
            cv2.imshow('img_close', img_close)
            cv2.waitKey(0)

        img_sub = img_close
        # #column wise threshold
        # img_col_wise_otsu_threshold = np.zeros(img_sub.shape, dtype=np.uint8)
        # img_col_wise_adoptive_threshold = np.zeros(img_sub.shape, dtype=np.uint8)
        #
        # for col in range(img_col_wise_otsu_threshold.shape[1]):
        #
        #     _, tmp = cv2.threshold(np.array(img_sub[:, col], dtype=np.uint8), 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #     img_col_wise_otsu_threshold[:, col] = tmp.reshape(1, img_sub.shape[0])
        #
        #     img_col_wise_adoptive_threshold = cv2.adaptiveThreshold(np.array(img_sub, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
        #
        # if self.flag_imshow_on == 1:
        #     cv2.imshow('img_row_wise_otsu_threshold', np.concatenate((img_col_wise_otsu_threshold, img_col_wise_adoptive_threshold), axis=1))
        #     cv2.waitKey(0)
        #
        #     # start to opening the image(morpoholgy)
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        #     # np.zeros((300, img_hue.shape[1]), dtype=np.uint8)
        #     img_col_wise_adoptive_threshold = cv2.morphologyEx(img_col_wise_adoptive_threshold, cv2.MORPH_OPEN, kernel)
        #
        #     if self.flag_imshow_on == 1:
        #         cv2.namedWindow('img_col_wise_adoptive_threshold_opened')
        #         cv2.imshow('img_col_wise_adoptive_threshold_opened', img_col_wise_adoptive_threshold)
        #         cv2.waitKey(0)

        #original
        _, img_threshold = cv2.threshold(img_sub, 40, 255, cv2.THRESH_BINARY)

        _, img_threshold_otsu = cv2.threshold(np.array(img_sub, dtype=np.uint8), 10, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        imb_blur = cv2.GaussianBlur(img_sub, (5, 5), 0)
        _, img_threshold_otsu_blur = cv2.threshold(np.array(imb_blur, dtype=np.uint8), 30, 255,
                                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # # split -> otsh -> merge
        # middle_row = (img_sub.shape[0] // 2)
        # middle_col = (img_sub.shape[1] // 2)
        #
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_sub_otsu_1')
        #     cv2.imshow('img_sub_otsu_1', img_sub)
        #     cv2.waitKey(0)
        #
        # _, img_split_otsu = cv2.threshold(np.array(img_sub[:middle_row, :middle_col], dtype=np.uint8), 10, 255,
        #                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img_sub[:middle_row, :middle_col] = img_split_otsu
        #
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_sub_otsu_2')
        #     cv2.imshow('img_sub_otsu_2', img_sub)
        #     cv2.waitKey(0)
        #
        # _, img_split_otsu = cv2.threshold(np.array(img_sub[middle_row:, :middle_col], dtype=np.uint8), 10, 255,
        #                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img_sub[middle_row:, :middle_col] = img_split_otsu
        #
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_sub_otsu_3')
        #     cv2.imshow('img_sub_otsu_3', img_sub)
        #     cv2.waitKey(0)
        #
        # _, img_split_otsu = cv2.threshold(np.array(img_sub[:middle_row, middle_col:], dtype=np.uint8), 10, 255,
        #                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img_sub[:middle_row, middle_col:] = img_split_otsu
        #
        # _, img_split_otsu = cv2.threshold(np.array(img_sub[middle_row:, middle_col:], dtype=np.uint8), 10, 255,
        #                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img_sub[middle_row:, middle_col:] = img_split_otsu
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_sub_otsu')
        #     cv2.imshow('img_sub_otsu', img_sub)
        #     cv2.waitKey(0)






        # img_adaptive_threshold_mean = cv2.adaptiveThreshold(np.array(img_sub, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                                     cv2.THRESH_BINARY_INV, 11 ,2)
        # img_adaptive_threshold_gaussian = cv2.adaptiveThreshold(np.array(img_sub, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                                cv2.THRESH_BINARY_INV, 19, 2)

        #fucking code
        img_threshold_otsu[496:, :] = np.zeros((img_threshold_otsu.shape[0] - 496, img_threshold_otsu.shape[1]), dtype=np.uint8)
        img_threshold_otsu[317:440, 685:885] = np.zeros((123, 200), dtype=np.uint8)
        img_threshold_otsu[358:388, 896:934] = np.zeros((388 - 358, 934 - 896), dtype=np.uint8)

        img_whiteLine = np.zeros(img_threshold_otsu.shape, dtype=np.uint8)

        for i in range(img_threshold_otsu.shape[0]):
            for j in range(img_threshold_otsu.shape[1]):

                # copy the white line area
                img_whiteLine[i, j] = img_threshold_otsu[i, j]

                # Thresh_INV
                if img_threshold_otsu[i, j] == 0:
                    img_threshold_otsu[i, j] = 255
                else:
                    img_threshold_otsu[i ,j] = 0


        if self.flag_imshow_on == 1:
            # cv2.namedWindow('img_threshold')
            # cv2.imshow('img_threshold',
            #            np.concatenate((img_threshold, img_threshold_otsu, img_threshold_otsu_blur), axis=1))
            # img_adaptive_threshold_mean, img_adaptive_threshold_gaussian
            cv2.namedWindow('img_threshold_otsu')
            cv2.imshow('img_threshold_otsu', img_threshold_otsu)
            cv2.waitKey(0)

        # connected compnent labeling
        num_labels, img_connectedComponents = cv2.connectedComponents(image=img_threshold_otsu,
                                                                      connectivity=8)
        color = np.zeros((num_labels, 3), dtype=np.uint8)
        I_need_only_one_color = [180,180,0]  # [randint(0, 255), randint(0, 255), randint(0, 255)]
        for i in range(num_labels):
            color[i] = I_need_only_one_color

        img_label = np.zeros(frame.shape, dtype=np.uint8)
        for i in range(img_label.shape[0]):
            for j in range(img_label.shape[1]):
                if img_connectedComponents[i, j] > 1:# and img_connectedComponents[i, j] != 5:
                    # the reason " > 1 " is that we do know want background to be labeled.
                    img_label[i, j] = color[img_connectedComponents[i, j]]
                elif img_whiteLine[i, j] == 255:
                    img_label[i, j] = np.array([0, 255, 255], dtype=np.uint8)

        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_label')
            cv2.imshow('img_label', img_label)
            cv2.waitKey(0)

        # the reason is that we do know want background, red lines, non-parking area(label = 6) to be labeled.
        num_labels = num_labels - 3

        if self.flag_print_on == 1:
            print('what is the result', np.shape(img_connectedComponents), 'max is ',
                  np.max(img_connectedComponents))
            print('num_labels is ', num_labels)

        return img_label #img_threshold_otsu

    def redLineSegmentation(self, frame, pixelLeftTop=None, pixelRightBottom=None):

        if self.flag_imshow_on == 2:
            cv2.namedWindow('origin figure')
            if pixelRightBottom == None:
                cv2.imshow('origin figure', frame)
            else:
                cv2.imshow('origin figure after yolo', frame[pixelLeftTop[1]:pixelRightBottom[1]][pixelLeftTop[0]:pixelRightBottom[0]][:])
            cv2.waitKey(1)


        ##################################################################################

        # convert rgb to hsv
        # img_hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]

        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('hue_figure')
        #     cv2.imshow('hue_figure', img_hue)
        #     cv2.waitKey(0)
        
        if pixelLeftTop == None: #cannot detect the car with YOLO
            img_saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        else:
            img_saturation = cv2.cvtColor(frame[pixelLeftTop[1]:pixelRightBottom[1]][pixelLeftTop[0]:pixelRightBottom[0]][:], cv2.COLOR_BGR2HSV)[:, :, 1]

        if self.flag_imshow_on == 2:
            cv2.namedWindow('saturation_figure')
            cv2.imshow('saturation_figure', img_saturation)
            cv2.waitKey(1)

        # img_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('value_figure')
        #     cv2.imshow('value_figure', img_value)
        #     cv2.waitKey(0)

        ##################################################################################








        ###############################                                    ################################

        # # threshold
        # # _, img_threshold_otsu = cv2.threshold(np.array(img_open, dtype=np.uint8), 80, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, img_threshold = cv2.threshold(img_saturation, 80, 255, cv2.THRESH_BINARY)
        # # imb_blur = cv2.GaussianBlur(img_close, (19, 19), 0)
        # # _, img_threshold_otsu_blur = cv2.threshold(np.array(imb_blur, dtype=np.uint8), 80, 255,
        # #                                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('cv2.THRESH_BINARY')
        #     cv2.imshow('cv2.THRESH_BINARY', img_threshold)
        #     cv2.waitKey(0)
        #
        #
        #
        #
        #
        # # # start to opening the image(morpoholgy)
        # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        # img_open = cv2.morphologyEx(img_saturation, cv2.MORPH_OPEN, kernel_open)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_open')
        #     cv2.imshow('img_open', img_open)
        #     cv2.waitKey(0)
        #
        # # threshold
        # # _, img_threshold_otsu = cv2.threshold(np.array(img_open, dtype=np.uint8), 80, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, img_threshold = cv2.threshold(img_open, 80, 255, cv2.THRESH_BINARY)
        # # imb_blur = cv2.GaussianBlur(img_close, (19, 19), 0)
        # # _, img_threshold_otsu_blur = cv2.threshold(np.array(imb_blur, dtype=np.uint8), 80, 255,
        # #                                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('cv2.THRESH_BINARY img_open')
        #     cv2.imshow('cv2.THRESH_BINARY img_open', img_threshold)
        #     cv2.waitKey(0)





        # start to closing the image(morpoholgy)
        global kernelSize_close
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize_close)
        img_close = cv2.morphologyEx(img_saturation, cv2.MORPH_CLOSE, kernel_close)
        if self.flag_imshow_on == 2:
            cv2.namedWindow('img_close (30, 30)')
            cv2.imshow('img_close (30, 30)', img_close)
            cv2.waitKey(1)

        # threshold
        # _, img_threshold_otsu = cv2.threshold(np.array(img_close, dtype=np.uint8), 80, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, img_threshold = cv2.threshold(img_close, 120, 255, cv2.THRESH_BINARY)
        if self.flag_imshow_on == 2:
            cv2.namedWindow('cv2.THRESH_BINARY')
            cv2.imshow('cv2.THRESH_BINARY', img_threshold)
            cv2.waitKey(1)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('cv2.THRESH_BINARY_OTSU img_close (30, 30)')
        #     cv2.imshow('cv2.THRESH_BINARY_OTSU img_close (30, 30)', img_threshold_otsu)
        #     cv2.waitKey(0)


        ###############################                                    ################################






        # start to opening the image(morpoholgy)
        # for i in [5, 10, 15, 20]:
        #     kernel_size = (i, i)
        #     kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        #     img_open = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel_open)
        #     if self.flag_imshow_on == 1:
        #         cv2.namedWindow('img_open' + str(kernel_size))
        #         cv2.imshow('img_open' + str(kernel_size), img_open)
        #         cv2.waitKey(0)
        global kernelSize_open
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize_open)
        img_open = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel_open)
        if self.flag_imshow_on == 2:
            cv2.namedWindow('img_open' + str(kernelSize_open))
            cv2.imshow('img_open' + str(kernelSize_open), img_open)
            cv2.waitKey(1)

        return img_open.copy()

    def startDrawCircle(self, frame, img_open):
        # cv2.imshow('fuck', img_open)
        # cv2.waitKey(1)
        #connected compnent labeling
        num_labels, img_connectedComponents, stat, centroid = cv2.connectedComponentsWithStats(image=img_open, connectivity=8, ltype=cv2.CV_16U)

        # print('num_labels are', num_labels)
        # print('stat is ', stat)
        # print('centroid is ', centroid)


        # for height in range(img_connectedComponents.shape[0]):
        #     for width in range(img_connectedComponents.shape[1]):





        ###################################################################################################################################################

        for label in range(1, num_labels): #label == 0 is background
            cv2.circle(frame, tuple(np.array(centroid[label, :], np.uint16)), radius=10, color=boxColor[label], thickness=3)
            # print('label is ', label)
            # cv2.imshow('find centroid', frame)
            # cv2.waitKey(0)
        #
        #     for i in range(frame.shape[0]): #shape = [height, width, depth]
        #         if img_connectedComponents[i, stat[label, 0]] == label:
        #             left_front_pixel = (stat[label, 0], i) # [height, widht] -> (widht, height)
        #             break
        #
        #     for j in range(frame.shape[1]): #shape = [height, width, depth]
        #         if img_connectedComponents[stat[label, 1], j] == label:
        #             right_front_pixel = (j, stat[label, 1])
        #             break
        #
        #     centroid_front_pixel = ((left_front_pixel[0] + right_front_pixel[0])//2, (left_front_pixel[1] + right_front_pixel[1])//2)
        # https: // docs.opencv.org / 3.0 - beta / modules / imgproc / doc / drawing_functions.html
        #     cv2.arrowedLine(img=frame, pt1=tuple(np.array(centroid[label, :], np.uint16)), pt2=centroid_front_pixel, color=(255, 255, 50), thickness=3) #, line_type=None, shift=None, tipLength=None
        #     cv2.circle(frame, left_front_pixel, radius=10, color=(0, 255, 255), thickness=3) #bgr type
        #     cv2.circle(frame, right_front_pixel, radius=10, color=(0, 0, 255), thickness=3)

        # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        # https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
        # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
        _, contours, _ = cv2.findContours(image=img_open, mode=1, method=2) #, hierarchy
        ######### LIST out of index ERROR : might be no .. car detected ... ##################################################################
        cnt = contours[0] ##   0 ##########################################################################3 fucking code ##################
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)



        global flag_1_measureONLY_2_kalman, boxColor
        if flag_1_measureONLY_2_kalman == 1:
            for i in range(box.shape[0]):
                cv2.circle(frame, tuple(box[i]), radius=5, color=tuple(boxColor[i]), thickness=2)

        elif flag_1_measureONLY_2_kalman == 2:

            # # for comparison with measurementONLY
            # for i in range(box.shape[0]):
            #     cv2.circle(frame, tuple(box[i]), radius=5, color=tuple(boxColor[i]), thickness=2)

            if self.flag_first_init_kalman == 1:

                self.kalmanInst = [[KalmanFilterClass(), KalmanFilterClass()] for i in range(4)] # 4 : 4 corners at the car
                # print('shape of the self.kalmanInst is', np.shape(self.kalmanInst))
                # detail of index of 4by2 matrix
                # [0, 0] : self.kalmanInst_width_front_right = KalmanFilterClass()
                # [0, 1] : self.kalmanInst_height_front_right = KalmanFilterClass()
                # [1, 0] : self.kalmanInst_width_front_left = KalmanFilterClass()
                # [1, 1] : self.kalmanInst_height_front_left = KalmanFilterClass()
                # [2, 0] : self.kalmanInst_width_rear_right = KalmanFilterClass()
                # [2, 1] :self.kalmanInst_height_rear_right = KalmanFilterClass()
                # [3, 0] : self.kalmanInst_width_rear_left = KalmanFilterClass()
                # [3, 1] :self.kalmanInst_height_rear_left = KalmanFilterClass()

                #find the front of the car. sometimes box[] is not correctly correspondent to what I meant.
                boxClosestIndex = [0, 1, 2, 3]
                boxIndex_in_order_of_kalman_index = np.zeros(4, dtype=np.uint8)
                self.fineClosestCorner(box, boxClosestIndex, boxIndex_in_order_of_kalman_index)


                for corner in boxIndex_in_order_of_kalman_index: #range(4):
                    for width_or_height in range(2):
                        self.kalmanInst[corner][width_or_height].initialMeasurement(z=box[corner, width_or_height])

                    # draw circle
                    cornerPixel = tuple((self.kalmanInst[corner][0].x_post[0], self.kalmanInst[corner][1].x_post[0]))
                    # print('cornerPixel is ', cornerPixel)
                    cv2.circle(frame, cornerPixel, radius=5, color=boxColor[corner], thickness=2)

                # cv2.imshow('boxIndex check', frame)
                # cv2.waitKey(0)

                #centroid kalman filter
                self.kalmanInstCentroid = [KalmanFilterClass(), KalmanFilterClass()]
                for width_or_height in range(2):
                    self.kalmanInstCentroid[width_or_height].initialMeasurement(z=centroid[1, width_or_height]) #  3  ######################################### should change '3' .. this is just for indicating that label(==3 is the car)

                #arrow kalman filter
                self.kalmanInstEndPixelOfArrow = [KalmanFilterClass(), KalmanFilterClass()]

                # we assume that the end_of_arrow_pixel has smaller height value than the height value of the centroid.
                # this is because we know where the car start in the project.
                self.which_corner_defines_end_of_arrow_pixel = np.zeros(2, dtype=np.int32) #[label00, label01]
                self.end_of_arrow_pixel = np.zeros(2, dtype=np.int32) #[[label00_width, label00_height], [label01_width, label01_height]]

                #car go straight at the first time in cctvView
                for corner in range(4):
                    if self.kalmanInst[corner][1].x_post[0] < centroid[1, 1]:  #  3  ######################################### should change '3' .. this is just for indicating that label(==3 is the car)
                        which_corner_defines_end_of_arrow_pixel = corner
                        break

                for width_or_height in range(2):
                    self.end_of_arrow_pixel[width_or_height] = int((self.kalmanInst[which_corner_defines_end_of_arrow_pixel][width_or_height].x_post[0]
                                                                    + self.kalmanInst[boxClosestIndex[which_corner_defines_end_of_arrow_pixel]][width_or_height].x_post[0])//2)

                    # origin_of_arrow_pixel is centroid of the car
                    self.kalmanInstEndPixelOfArrow[width_or_height].initialMeasurement(z=self.end_of_arrow_pixel[width_or_height])

                # # draw arrow
                # origin_of_arrow = tuple(np.stack((self.kalmanInstCentroid[0].x_post[0], self.kalmanInstCentroid[1].x_post[0])))
                # end_of_arrow = tuple(np.stack((self.kalmanInstEndPixelOfArrow[0].x_post[0], self.kalmanInstEndPixelOfArrow[1].x_post[0])))
                # cv2.arrowedLine(img=frame, pt1=origin_of_arrow, pt2=end_of_arrow, color=(255, 255, 50), thickness=3)  # , line_type=None, shift=None, tipLength=None

                #do not come here again
                self.flag_first_init_kalman = 0

            ## second iterative kalman filter
            else:
                for corner in range(4):
                    # print('for ', corner, '-th corner')
                    for width_or_height in range(2):
                        # print('width_or_height is ', width_or_height)

                        # box[width, height] is the corner pixel at the rectangle
                        min = np.sqrt(np.power((self.kalmanInst[corner][0].z_prior - box[0, 0]), 2) + np.power((self.kalmanInst[corner][1].z_prior - box[0, 1]), 2))
                        minIndex = 0
                        for label in range(1, box.shape[0]):
                            tmp = np.sqrt(np.power((self.kalmanInst[corner][0].z_prior - box[label, 0]), 2)
                                          + np.power((self.kalmanInst[corner][1].z_prior - box[label, 1]), 2))
                            if min > tmp:
                                min = tmp
                                minIndex = label

                        # print('min and minIndex is', min, minIndex)
                        self.kalmanInst[corner][width_or_height].measurementUpdate(z=np.array((box[minIndex, width_or_height]), dtype=np.float32))

                    #draw circle
                    cornerPixel = tuple((self.kalmanInst[corner][0].x_post[0], self.kalmanInst[corner][1].x_post[0]))
                    cv2.circle(frame, cornerPixel, radius=5, color=boxColor[corner], thickness=2)

                # find the front of the car. sometimes box[] is not correctly correspondent to what I meant.

                boxClosestIndex = [0, 1, 2, 3]
                boxIndex_in_order_of_kalman_index = np.zeros(4, dtype=np.uint8)
                self.fineClosestCorner(box, boxClosestIndex, boxIndex_in_order_of_kalman_index)

                # end_of_arrow kalman filter
                for width_or_height in range(2):
                    tmp = np.zeros(2, dtype=np.float32)
                    tmp[width_or_height] = int((self.kalmanInst[0][width_or_height].x_post[boxIndex_in_order_of_kalman_index[0]]
                                                + self.kalmanInst[boxIndex_in_order_of_kalman_index[1]][width_or_height].x_post[0]) // 2)
                    tmptmp = np.zeros(2, dtype=np.float32)
                    tmptmp[width_or_height] = int((self.kalmanInst[boxIndex_in_order_of_kalman_index[2]][width_or_height].x_post[0]
                                                   + self.kalmanInst[boxIndex_in_order_of_kalman_index[3]][width_or_height].x_post[0]) // 2)

                    dist_tmp_to_arrow = np.power(self.end_of_arrow[0] - tmp[0], 2) + np.power(self.end_of_arrow[1] - tmp[1], 2)
                    dist_tmptmp_to_arrow = np.power(self.end_of_arrow[0] - tmptmp[0], 2) + np.power(self.end_of_arrow[1] - tmptmp[1], 2)

                    #compare with the value 'self.end_of_arro' in prior state.
                    if dist_tmp_to_arrow > dist_tmptmp_to_arrow:
                        self.end_of_arrow = tmptmp
                    else:
                        self.end_of_arrow = tmp

                    self.kalmanInstEndPixelOfArrow[width_or_height].measurementUpdate(z=self.end_of_arrow_pixel[width_or_height])

                # centroid kalman filter,
                for width_or_height in range(2):
                    sum = 0
                    for corner in range(4):
                        sum = sum + self.kalmanInst[corner][width_or_height].x_post[0]
                    self.kalmanInstCentroid[width_or_height].measurementUpdate(z=sum // 4)


            #draw arrow
            # origin_of_arrow = tuple(np.stack((self.kalmanInstCentroid[0].x_post[0], self.kalmanInstCentroid[1].x_post[0])))
            origin_of_arrow = tuple(np.concatenate((self.kalmanInstCentroid[0].x_post[0], self.kalmanInstCentroid[1].x_post[0])))
            # self.end_of_arrow = tuple(np.concatenate((self.kalmanInstEndPixelOfArrow[0].x_post[0], self.kalmanInstEndPixelOfArrow[1].x_post[0])))
            self.end_of_arrow = tuple(np.stack((int(self.kalmanInstEndPixelOfArrow[0].x_post[0]), int(self.kalmanInstEndPixelOfArrow[1].x_post[0]))))
            # print('fuck this tuple thing : self.end_of_arrow is ', self.end_of_arrow, 'origin_of_arrow is', origin_of_arrow)
            cv2.arrowedLine(img=frame, pt1=origin_of_arrow, pt2=self.end_of_arrow, color=(255, 255, 50), thickness=3)  # , line_type=None, shift=None, tipLength=None

            self.theta = self.getTheta(origin_of_arrow)
            cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 50), thickness=3)

            #I want to send self.theta with type 'int'
            self.theta = int(self.theta)

                # for label in range(1, num_labels):  # label == 0 is background
                #     cv2.circle(frame, tuple(np.array(centroid[label, :], np.uint16)), radius=10, color=(255, 0, 150), thickness=3)
                #
                #     origin_of_arrow = tuple(np.array(centroid[label, :], np.uint16))
                #     self.end_of_arrow = (origin_of_arrow[0] + stat[label, 2], origin_of_arrow[1] + stat[label, 3])
                #     cv2.arrowedLine(img=frame, pt2=origin_of_arrow, pt1=self.end_of_arrow, color=(255, 255, 50), thickness=3)  # , line_type=None, shift=None, tipLength=None
                #     cv2.circle(frame, origin_of_arrow, radius=10, color=(0, 255, 255), thickness=3)  # bgr type
                #     cv2.circle(frame, self.end_of_arrow, radius=10, color=(0, 0, 255), thickness=3)

        # dst = cv2.cornerHarris(src=np.array(img_open, dtype=np.float32), blockSize=20, ksize=21, k=0.1)
        # # find centroids
        # frame[dst > 0.01 * dst.max()] = [0, 0, 255]




        # cv2.cornerEigenValsAndVecs
        # dst = cv2.cornerHarris(src=np.array(img_open, dtype=np.float32), blockSize=20, ksize=21, k=0.1)
        # # dst = cv2.dilate(dst, None)
        # ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        # dst = np.uint8(dst)
        #
        # # find centroids
        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.array(dst, np.uint8))
        #
        # # define the criteria to stop and refine the corners
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # corners = cv2.cornerSubPix(img_open, np.float32(centroids), (5, 5), (-1, -1), criteria)
        #
        # # Now draw them
        # res = np.hstack((centroids, corners))
        # res = np.int0(res)
        #
        # for corner in range(res.shape[0]):
        #     cv2.circle(frame, (res[corner, 0], res[corner, 1]), radius=3, color=(0, 0, 255), thickness=3)
        #     cv2.circle(frame, (res[corner, 2], res[corner, 3]), radius=3, color=(0, 255, 0), thickness=3)
        # # frame[res[:, 1], res[:, 0]] = [0, 0, 255]
        # # frame[res[:, 3], res[:, 2]] = [0, 255, 0]




        ###################################################################################################################################################


        if self.flag_imshow_on == 2:
            # print('result of the redLineSeg is ', self.end_of_arrow, self.theta)
            cv2.namedWindow('redLineSeg')
            # cv2.imshow('redLineSeg', np.concatenate((frame[:, :, 1], img_open), axis=1))
            cv2.imshow('redLineSeg', frame)
            cv2.waitKey(1)

            # for hello in range(10):
            #     print(' ')


        return frame, self.end_of_arrow, self.theta
    def fineClosestCorner(self, box, boxClosestIndex, boxIndex_in_order_of_kalman_index):
        boxIndex =[0, 1, 2, 3]
        for corner in boxIndex:
            # print('corner=',corner)
            dist_corner_to_corner_min = 99999
            for corner_to_check in range(4):
                # print('corner_to_check', corner_to_check)
                if corner != corner_to_check:
                    dist_corner_to_corner = 0
                    for width_or_height in range(2):
                        dist_corner_to_corner = dist_corner_to_corner + np.power(box[corner][width_or_height] - box[corner_to_check][width_or_height], 2)

                    if dist_corner_to_corner_min > dist_corner_to_corner:
                        dist_corner_to_corner_min = dist_corner_to_corner
                        boxClosestIndex[corner] = corner_to_check
                        # print('corner=',corner,'corner_to_check=',corner_to_check,'boxClosestIndex',boxClosestIndex)
                        # print('dist_corner_to_corner',dist_corner_to_corner)

        # print('boxIndex = ', boxIndex, 'boxClosestIndex = ', boxClosestIndex)
        boxIndex_in_order_of_kalman_index[1] = boxClosestIndex[0]
        if boxIndex_in_order_of_kalman_index[1] != 1:
            boxIndex_in_order_of_kalman_index[2] = 1
            boxIndex_in_order_of_kalman_index[3] = boxClosestIndex[1]
        else:
            boxIndex_in_order_of_kalman_index[2] = 2
            boxIndex_in_order_of_kalman_index[3] = boxClosestIndex[2]
        print('boxIndex_in_order_of_kalman_index = ', boxIndex_in_order_of_kalman_index)

    def getTheta(self, origin_of_arrow):
        # calculate the theta
        horizontal_length = (self.end_of_arrow[0] - origin_of_arrow[0])
        vertical_length = (self.end_of_arrow[1] - origin_of_arrow[1])
        if horizontal_length >= 0 and vertical_length >= 0:  # range : 0 ~ pi/2
            theta = np.arctan(horizontal_length / vertical_length) + pi
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=boxColor[0], thickness=3)

        elif horizontal_length >= 0 and vertical_length < 0:  # range : pi/2 ~ pi
            theta = np.arctan(horizontal_length / vertical_length) + (2 * pi)
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=boxColor[1], thickness=3)

        elif horizontal_length < 0 and vertical_length < 0:
            theta = np.arctan(horizontal_length / vertical_length)
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=boxColor[2], thickness=3)

        elif horizontal_length < 0 and vertical_length >= 0:  # - pi/2 ~ 0
            theta = np.arctan(horizontal_length / vertical_length) + pi
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=boxColor[3], thickness=3)

        # print(' >> self.theta, horizontal_length, vertical_length is  ', self.theta, horizontal_length, vertical_length)
        return (theta * 180 / pi)

    def fuck(self):

        # dilate image
        # for img in [img_threshold_otsu_blur]: #img_threshold, img_threshold_otsu,
        # blur makes small noise gone
        # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # img_dilate = cv2.morphologyEx(img_close, cv2.MORPH_DILATE, kernel_dilate)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_dilate')
        #     cv2.imshow('img_dilate', img_dilate)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()



        # threshold
        # _, img_threshold_otsu = cv2.threshold(np.array(img_open, dtype=np.uint8), 80, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, img_threshold = cv2.threshold(img_open, 80, 255, cv2.THRESH_BINARY)
        imb_blur = cv2.GaussianBlur(img_close, (19, 19), 0)
        _, img_threshold_otsu_blur = cv2.threshold(np.array(imb_blur, dtype=np.uint8), 80, 255,
                                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if self.flag_imshow_on == 1:
            # cv2.namedWindow('img_threshold')
            # cv2.imshow('img_threshold',
            #            np.concatenate((img_threshold, img_threshold_otsu, img_threshold_otsu_blur), axis=1))
            # cv2.imshow('cv2.THRESH_BINARY', img_threshold)
            # cv2.imshow('cv2.THRESH_BINARY + cv2.THRESH_OTSU', img_threshold_otsu)
            cv2.namedWindow('cv2.THRESH_BINARY + cv2.THRESH_OTSU + blur')
            cv2.imshow('cv2.THRESH_BINARY + cv2.THRESH_OTSU + blur', img_threshold_otsu_blur)
            cv2.waitKey(0)

        num_labels, img_connectedComponents = cv2.connectedComponents(image=img_threshold_otsu_blur, connectivity=8)


        color = np.zeros((num_labels, 3), dtype=np.uint8)

        I_need_only_one_color = [255, 0, 255]#[randint(0, 255), randint(0, 255), randint(0, 255)]
        for i in range(num_labels):
            color[i] = I_need_only_one_color

        img_label = np.zeros(frame.shape, dtype=np.uint8)
        for i in range(img_label.shape[0]):
            for j in range(img_label.shape[1]):
                if img_connectedComponents[i, j] > 1 and img_connectedComponents[i, j] != 6:
                    # the reason " > 1 " is that we do know want background to be labeled.
                    img_label[i, j] = color[img_connectedComponents[i, j]]

        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_label')
            cv2.imshow('img_label', img_label)
            cv2.waitKey(0)

        # the reason is that we do know want background, red lines, non-parking area(label = 6) to be labeled.
        num_labels = num_labels - 3

        if self.flag_print_on == 1:
            print('what is the result', np.shape(img_connectedComponents), 'max is ',
                  np.max(img_connectedComponents))
            print('num_labels is ', num_labels)

        #alpha blending
        alpha = 0.5
        frame = cv2.addWeighted(src1=frame, alpha=alpha, src2=img_label, beta=(1-alpha), gamma=0.0,
                                        dst=None, dtype=-1)
        if self.flag_imshow_on == 1:
            cv2.namedWindow(str('alphaBlended alpha = ' + str(alpha)))
            cv2.imshow(str('alphaBlended alpha = ' + str(alpha)), frame)
            cv2.waitKey(0)

        return frame

    def cornerDetect(self, frame, mask=None):
        print('>> into the corenrDetect')
        # harris corner detector
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
        print('type of the source is', frame.dtype, 'shape is ', frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(src=np.array(gray, dtype=np.float32), blockSize=10, ksize=3, k=0.1)
        # as k is larger, detector becomes robust to noise
        # dst has the same shape of the src.
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(np.array(gray, dtype=np.float32), np.float32(centroids), (5, 5), (-1, -1), criteria)

        if self.flag_print_on == 1:
            print('shape of the corner is ', corners.shape)#'corner is ', corners,
            print('shape of the image', frame.shape)
            # print('what is dst', np.sort(dst), 'shape of the dst is ', dst.shape)
            # print('waht the fuck', dst > 0.01 * dst.max())


        # Now draw them
        # frame[dst > 0.01 * dst.max()] = [0, 0, 255]
        res = np.hstack((centroids, corners, (np.arange(corners.shape[0])).reshape(-1,1)))
        res = np.array(res, dtype=np.uint)
        if self.flag_print_on == 1:
            print('res is after np.array')
            self.myPrint(res)

        res_sorted = np.array(sorted(res, key=lambda res: res[2]), dtype=np.uint)
        #res_sorted = np.sort(res, order='x2')
        if self.flag_print_on == 1:
            print('sorted res is ')
            self.myPrint(res_sorted)

        index_center = np.where(res_sorted[:, 4] == 0)
        print('index_center is ')
        self.myPrint(index_center)

        # first localize the out corners. Then calculate the distance from the vector of each two outliers.
        # left_out_courner = res_sorted[(index_center-2) ]


        num_of_src_points = 14 * 2
        src_point = np.zeros((num_of_src_points, 2), dtype=np.uint)
        for i in range(num_of_src_points):
            # half on the left side, the others in right side.
            if i >= (num_of_src_points / 2):
                # do not include center because it is not a corner just symbol.
                src_point[i] = res_sorted[(i + index_center[0] - (num_of_src_points / 2) + 1), 2:(3 + 1)]
            else:
                src_point[i] = res_sorted[(i + index_center[0] - (num_of_src_points / 2)), 2:(3+1)]

        if self.flag_print_on == 1:
            print('src_point', src_point)
            #self.myPrint(src_point)
        # cv2.imshow('result of the corner detection', frame)
        # cv2.waitKey(0)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> fuck')
        if self.flag_imshow_on == 1:

            # draw circles at the corners
            print('res.shape[0] is ', res.shape[0])
            for i in range(0, res.shape[0]):
                # I do not know why res[0, :] is the center of the image..??
                cv2.circle(frame, (res[i, 0], res[i, 1]), 5, (0, 0, 255))
                cv2.circle(frame, (res[i, 2], res[i, 3]), 5, (0, 255, 0))

            print('num_of_src_points are', num_of_src_points)
            for i in range(num_of_src_points):
                src_point = tuple(map(tuple, src_point))
                cv2.circle(frame, src_point[i], 5, (255, 255, 0), thickness=-1)

            cv2.imshow('result of the corner detection', frame)
            cv2.waitKey(0)





        # Hough Line detection
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        # minLineLength = 100
        # maxLineGap = 10
        #
        # edges = cv2.Canny(img_dilate, 50, 150, apertureSize=3)
        # print('shape of the frame is ', img_dilate.shape)
        # print('shape of the edges is ', edges.shape, 'edges is ', edges, 'max is ', np.max(edges))
        #
        # mask_corner = np.zeros(img_dilate.shape[::-1], dtype=np.uint8)
        # for i in range(1, res.shape[0]):
        #     mask_corner[res[i,2], res[i,3]] = 255
        #
        #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        # print('what is the fucking lines', lines, 'shape of the lines', lines.shape)
        # for i in range(lines.shape[0]):
        #     for x1, y1, x2, y2 in lines[i]:
        #         cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=10)
        #         print('what is lines[', i, '] = ', lines[i])
        #
        #
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        # for i in range(lines.shape[0]):
        #     for rho, theta in lines[i]:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * (a))
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * (a))
        #         cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #
        # if self.flag_imshow_on == 1:
        #     cv2.imshow('canny', edges)
        #     cv2.imshow('result of the hough transform', frame)
        #     cv2.waitKey(0)

    def myPrint(self, array):
        array = np.array(array)
        print('print start and shape is ', array.shape)
        for i in range(array.shape[0]):
            print(array[i])

class calibClassCar:
    flag_first_didHomography = 1
    def __int__(self):
        print('calibClassCar init')

    def startHomography(self, frame, reverse=0):

        if self.flag_first_didHomography == 1:

            srcPixel_image = np.array([
                #[205, 946], [656 ,918], [1116, 936],
                            [207, 748], [435, 731], [653, 723], [875, 724], [1114, 732],
                [202, 684], [358, 672], [505, 663], [652, 659], [800, 658], [952, 600],
                [318, 643], [433, 636], [542, 631], [652, 628], [760, 625], [876, 626],
                [475, 615], [564, 612], [652, 609], [740, 607],
                [504, 600], [579, 599], [652, 595], [724, 594]

            ])

            dstPixel_ground = np.array([
                                      #[90, 225], [135, 225], [180, 225],
                           [45, 180], [90, 180], [135, 180], [180, 180], [225, 180],
                [0, 135],  [45, 135], [90, 135], [135, 135], [180, 135], [225, 135],
                [0, 90],   [45,  90], [90,  90], [135,  90], [180,  90], [225,  90],
                           [45,  45], [90,  45], [135,  45], [180,  45],
                           [45,   0], [90,   0], [135,   0], [180,   0]
            ])

            # for i in range(dstPixel_ground.shape[0]):
            #     dstPixel_ground[i][0] = dstPixel_ground[i][0] + 60

            for i in range(dstPixel_ground.shape[0]):
                dstPixel_ground[i][0] = dstPixel_ground[i][0] + 60
                dstPixel_ground[i][1] = dstPixel_ground[i][1] + 180

            # srcPixel_image, dstPixel_ground = np.array(srcPixel_image), np.array(dstPixel_ground)

            #img to ground homography
            if reverse == 0:
                self.homography_RANSAC, mask = cv2.findHomography(srcPoints=srcPixel_image,
                                                                    dstPoints=dstPixel_ground,
                                                                    method=cv2.RANSAC)
            #ground to img homogrphy
            elif reverse == 1:
                self.homography_RANSAC, mask = cv2.findHomography(dstPoints=srcPixel_image,
                                                                    srcPoints=dstPixel_ground,
                                                                    method=cv2.RANSAC)
            print('homography is ', self.homography_RANSAC)

            #do not calculate homography again
            self.flag_first_didHomography = 0

        if reverse == 0:
            frame_homography_RANSAC = cv2.warpPerspective(frame, self.homography_RANSAC, (360, 300 + 180))  # frame.shape[:2][::-1]
        else:
            frame_homography_RANSAC = cv2.warpPerspective(frame, self.homography_RANSAC, (1288, 964))  # frame.shape[:2][::-1]

        return frame_homography_RANSAC

    def alphaBlending(self, img1, img2):
        alpha = 0.3
        beta = (1 - alpha)
        return cv2.addWeighted(src1=img1, alpha=alpha, src2=img2, beta=beta, gamma=0.0, dst=None, dtype=-1)

class DataLoadClass:
    imgInst = []
    calibInst = []
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1
    flag_first_get_homography_ground_to_carView = 1

    def __init__(self, imgInst, calibInst):
        self.imgInst = imgInst
        self.calibInst = calibInst
        self.bridge = CvBridge()
        self.lineSegInst = LineSegClass(flag_imshow_on=1, flag_print_on=1)

        global flag_1_use_YOLO_2_not_use_YOLO
        if flag_1_use_YOLO_2_not_use_YOLO == 1:
            self.myDarknetInst = MyDarknet()
            self.myDarknetInst.__int__()
            print('self.myDarknetInst' , self.myDarknetInst)

    def subscribeImg(self):
        print('start to subscribe image')

        if flag_is_compressed_image == 0:
            # rospy.init_node('DataLoadClass', anonymous=True)
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
            # automatically go to the callback function : self.callback()

        elif flag_is_compressed_image == 1:
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, CompressedImage, self.callback, queue_size=1)

        else:
            print('flag_is_compressed_image is wrong')

    def loadImgInFolder(self):
        global fileList
        fileList = glob.glob(path_load_image)
        print('path_load_image is ', path_load_image)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        fileList = np.sort(fileList)
        print('what is fileList', fileList, '\n')

        global count
        for i in fileList:
            count = count + 1
            print('The ', count, '-th image is ', i)
            self.imgInst.saveImage(cv2.imread(i))

            self.wrapper(nameOfFile=i)
            
            # cv2.imshow(i, cv2.imread(i))
            # cv2.waitKey(0)
            # print('save the roi : _' + path_save_image + 'roi/' + i[-10:])
            # cv2.imwrite(str(path_save_image + 'roi/' + i[-10:]), cv2.resize(tmp[190:440, 260:510], (0,0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))

    def image2video(self):
        global fileList, num_of_image_in_database
        fileList = glob.glob(path_load_image)
        # print('path_image_database is ', str(save_concat_result + '*.png'))
        num_of_image_in_database = len(fileList)

        fileList = np.sort(fileList)
        self.flag_first_set_codec = 1

        for image in fileList:
            frame = cv2.imread(image)

            if self.flag_first_set_codec == 1:
                # set codec
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter((path_save_image + 'cctvView_yolo/' + 'output.avi'), fourcc, 5.0, frame.shape[:2][::-1])

                self.flag_first_set_codec = 0

            out.write(frame)

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

    def publishImg(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish JS image')
        self.rospyPubImg = rospy.Publisher('desktop/image_annotated', Image, queue_size=10)
        # rospy.Rate(10)  # 10Hz

    def serviceCarInfo(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish CarLocation, CarAngle')
        self.rospyServiceCarInfo = rospy.Service('desktop/carInfo', carInfo, self.carInfo_Request)

    def carInfo_Request(self, req):
        print('carInfo_Request : ', (self.carLocation[0], self.carLocation[1]), self.carAngle)
        return carInfoResponse(self.carLocation[0], self.carLocation[1], self.carAngle)


    def callback(self, data):
        global count
        try:
            count = count + 1

            if flag_is_compressed_image == 0:
                # parse message into image
                # bgr8: CV_8UC3, color image with blue-green-red color order and 8bit
                self.imgInst.saveImage(self.bridge.imgmsg_to_cv2(data, "bgr8"))

            elif flag_is_compressed_image == 1:
                np_arr = np.fromstring(data.data, np.uint8)
                self.imgInst.saveImage(cv2.imdecode(np_arr, cv2.IMREAD_COLOR))  # cv2.CV_LOAD_IMAGE_COLOR is out of version

            else:
                print('wrong flag_1_compressed_image... ')

            global flag_go_to_wrapper_or_save_subImage
            if flag_go_to_wrapper_or_save_subImage == 0:
                tmp = self.imgInst.imgData
                cv2.imwrite((path_save_image + 'cctvView/' + str((count + 10000)) + '.png'), tmp)
                cv2.imshow('hello', tmp)
                cv2.waitKey(1)

            elif flag_go_to_wrapper_or_save_subImage == 1:
                # if you want to work asynchronously, edit the lines below
                self.wrapper()

            else:
                print('wrong flag_go_to_wrapper_or_save_sub_Images')

        except CvBridgeError as e:
            print(e)


    def wrapper(self, nameOfFile=None):
        global count, flag_homography_to_drawCircle, flag_origin_to_homography, flag_1_use_YOLO_2_not_use_YOLO

        # cv2.imshow('check imgData', self.imgInst.imgData)
        # cv2.waitKey(0)

        #origin image to homography
        if flag_origin_to_homography == 1:
            self.imgInst.imgHomography = self.calibInst.startHomography(self.imgInst.imgData)
        else:
            self.imgInst.imgHomography = self.imgInst.imgData.copy()

        # cv2.imshow('check homography', self.imgInst.imgHomography)
        # cv2.waitKey(0)

        ##homography to drawCirble
        #not use yolo
        if flag_homography_to_drawCircle == 1 and flag_1_use_YOLO_2_not_use_YOLO == 2:
            self.imgInst.redLineSeg = self.lineSegInst.redLineSegmentation(self.imgInst.imgHomography)
            cv2.imshow('redLineSeg', self.imgInst.redLineSeg)
            cv2.waitKey(1)
            self.imgInst.imgDrawCircle, self.carLocation, self.carAngle = self.lineSegInst.startDrawCircle(frame=self.imgInst.imgHomography.copy(), img_open=self.imgInst.redLineSeg)

        #use yolo
        elif flag_homography_to_drawCircle == 1 and flag_1_use_YOLO_2_not_use_YOLO == 1:

            self.imgInst.imgYOLO, pixelLeftTop_result, pixelRightBottom_result, label_result = self.myDarknetInst.initDetection(self.imgInst.imgHomography.copy())

            # find the car in the cctvView
            car_index = -1
            label_index = 0
            for label in label_result:
                if label == 'cell phone' or label == 'car':
                    car_index = label_index
                    break
                else:
                    label_index = label_index + 1

            # spot the car convex hull
            if np.shape(label_result)[0] == 0 or car_index == -1: # if nothing was detected by YOLO, or there is no car in the image
                print('yolo did not spot the car')
                self.imgInst.redLineSeg = self.lineSegInst.redLineSegmentation(self.imgInst.imgHomography)

            else: #there is somthing detected as a car
                print('yolo spot the car')
                # cv2.imshow('result of yolo', self.imgInst.imgYOLO)
                # cv2.waitKey(1)
                self.imgInst.redLineSeg = np.zeros(self.imgInst.imgHomography.shape[:2][::-1], dtype=np.float32)
                self.imgInst.redLineSeg[pixelLeftTop[1]:pixelRightBottom[1]][pixelLeftTop[0]:pixelRightBottom[0]]=\
                    self.lineSegInst.redLineSegmentation(self.imgInst.imgHomography,
                                                         pixelLeftTop=pixelLeftTop_result[car_index],
                                                         pixelRightBottom=pixelRightBottom_result[car_index])
            # extract car information
            self.imgInst.imgDrawCircle, self.carLocation, self.carAngle = self.lineSegInst.startDrawCircle(frame=self.imgInst.imgHomography.copy(),
                                                                                                        img_open=self.imgInst.redLineSeg)

        else:
            self.imgInst.redLineSeg = self.imgInst.imgHomography.copy()
            self.imgInst.imgDrawCircle = self.imgInst.imgHomography.copy()
            self.carLocation = (0, 0)
            self.carAngle = 0

        if flag_drawCircle_to_carView == 1:
            if self.flag_first_get_homography_ground_to_carView == 1:
                self.calibClassCar_inst = calibClassCar()

                #do not come here again
                self.flag_first_get_homography_ground_to_carView = 2

            matrixRotation = cv2.getRotationMatrix2D(center=self.carLocation, angle=360 - self.carAngle, scale=1)
            self.imgInst.imgSeeThrough = cv2.warpAffine(src=self.imgInst.imgDrawCircle, M=matrixRotation, dsize=self.imgInst.imgDrawCircle.shape[:2][::-1])

            cameraLocation_in_carView = (199, 461)  ############### fuck ############################################################
            tmp = (cameraLocation_in_carView[0] - self.carLocation[0], cameraLocation_in_carView[1] - self.carLocation[1])
            matrixTranslate = np.float32([[1, 0, tmp[0]], [0, 1, tmp[1]]])
            self.imgInst.imgSeeThrough = cv2.warpAffine(src=self.imgInst.imgSeeThrough, M=matrixTranslate, dsize=self.imgInst.imgSeeThrough.shape[:2][::-1])

            self.imgInst.imgSeeThrough = self.calibClassCar_inst.startHomography(frame=self.imgInst.imgSeeThrough, reverse=1)
            global path_carView_undistort_image
            self.carView_undistort = cv2.imread(path_carView_undistort_image + str((100000 + count)) + '.png')
            self.imgInst.imgSeeThrough = self.calibClassCar_inst.alphaBlending(img1=self.imgInst.imgSeeThrough, img2=self.carView_undistort)

            cv2.imshow('self.imgInst.imgSeeThrough', self.imgInst.imgSeeThrough)
            cv2.waitKey(1)



        ##save frames
        if flag_saveImg == 1:
            if flag_1_subscribeImg_2_loadImgFile_3_image2video == 2:
                # cv2.imwrite(path_save_image + 'cctvView/' + nameOfFile[-10:], self.imgInst.imgData)

                if flag_origin_to_homography == 1:
                    print('save iamge with name : ', path_save_image + 'cctvView_homography/' + nameOfFile[-10:])
                    cv2.imwrite(path_save_image + 'cctvView_homography/' + nameOfFile[-10:], self.imgInst.imgHomography)

                if flag_1_use_YOLO_2_not_use_YOLO == 1:
                    cv2.imwrite(path_save_image + 'cctvView_yolo/' + nameOfFile[-10:], self.imgInst.imgYOLO)

                if flag_homography_to_drawCircle == 1:
                    print('save iamge with name : ', path_save_image + 'cctvView_redLineSeg/' + nameOfFile[-10:])
                    cv2.imwrite(path_save_image + 'cctvView_redLineSeg/' + nameOfFile[-10:], self.imgInst.redLineSeg)
                    cv2.imwrite(path_save_image + 'cctvView_center/' + nameOfFile[-10:], self.imgInst.imgDrawCircle)

                if flag_drawCircle_to_carView == 1:
                    print('save iamge with name : ', path_save_image + 'cctvView_2_carView/' + nameOfFile[-10:])
                    cv2.imwrite(path_save_image + 'cctvView_2_carView/' + nameOfFile[-10:], self.imgInst.imgSeeThrough)

            elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 1:
                cv2.imwrite(path_save_image + 'cctvView/' + str(100000 + count) + '.png', self.imgInst.imgData)
                print('save iamge with name : ', path_save_image + 'cctvView_homography/' + str(100000 + count) + '.png')
                cv2.imwrite((path_save_image + 'cctvView_homography/' + str(100000 + count) + '.png'), self.imgInst.imgHomography)

                if flag_1_use_YOLO_2_not_use_YOLO == 1:
                    cv2.imwrite((path_save_image + 'cctvView_yolo/'+ str(100000 + count) + '.png'), self.imgInst.imgYOLO)

                if flag_homography_to_drawCircle == 1:
                    print('save iamge with name : ', path_save_image + 'cctvView_redLineSeg/'+ str(100000 + count) + '.png')
                    cv2.imwrite((path_save_image + 'cctvView_redLineSeg/'+ str(100000 + count) + '.png'), self.imgInst.redLineSeg)
                    cv2.imwrite((path_save_image + 'cctvView_center/'+ str(100000 + count) + '.png'), self.imgInst.imgDrawCircle)

                if flag_DrawCircle_to_carView == 1:
                    print('save iamge with name : ', path_save_image + 'cctvView_2_carView/' + nameOfFile[-10:])
                    cv2.imwrite(path_save_image + 'cctvView_2_carView/' + nameOfFile[-10:], self.imgInst.imgSeeThrough)

        #re-publish processed images
        if flag_publishImg == 1:
            try:
                self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(self.imgInst.imgDrawCircle, "bgr8"))
                # self.rospyServiceCarInfo.publish((str(carLocation) + ',' + str(carAngle)))
            except CvBridgeError as e:
                print(e)
        else:
            cv2.imshow('self.imgInst.imgDrawCircle', self.imgInst.imgDrawCircle)
            cv2.waitKey(1)

class ImgClass:
    height = 0
    width = 0
    imgData = None
    imgUndistort = None
    imgHomography = None
    imgYOLO = None
    imgRedLineSeg = None
    imgDrawCircle = None
    imgSeeThrough = None

    def saveImage(self, img):
        # if (964 != img.shape[0]) or (1288 != img.shape[1]): #(1288, 964) != img.shape[:2][::-1]
        #     print('img.shape[:2]', img.shape[:2])
        #     cv2.imshow('saveImage', img)
        #     cv2.waitKey(1)
        #
        #     print('begin reshape the self.imgInst.imgData')
        #     self.imgData = cv2.resize(src=img, dsize=(1288,964))
        #     # cv2.imshow('saveImage', self.imgData)
        #     # cv2.waitKey(1)
        # else:
        self.imgData = img
        self.height, self.width = self.imgData.shape[:2]

class Import_cctvView:
    end_of_array = np.zeros(2, dtype=np.float32)

    def __int__(self):
        print('__init__ in the class < Import_cctvView >')
        self.imgInst = ImgClass()
        self.calibInst = CalibClass()
        self.dataLoadInst = DataLoadClass(self.imgInst, self.calibInst)
        self.lineSegInst = LineSegClass(flag_imshow_on=1, flag_print_on=1)

        global flag_origin_to_homography, flag_homography_to_drawCircle
        flag_origin_to_homography = 1
        flag_homography_to_drawCircle = 1

        global flag_1_fisheye_2_non_fisheye, flag_1_measureONLY_2_kalman
        flag_1_fisheye_2_non_fisheye = 1
        flag_1_measureONLY_2_kalman = 2

        flag_is_compressed_image = 0
        flag_go_to_wrapper_or_save_subImage = 1

    def getCarLocationAngle(self, frame_cctv=None):
        # _, end_of_array, angle = self.lineSegInst.redLineSegmentation(frame_cctv)
        # return end_of_array, angle

        # subscribe cctv images
        if frame_cctv is None:
            dataLoadInst.serviceCarInfo()

        #load cctv images in the folders
        else:
            return self.lineSegInst.startDrawCircle(frame=frame_cctv, img_open=self.lineSegInst.redLineSegmentation(frame_cctv))




if __name__ == "__main__":

    print("check the version opencv.")
    print(cv2.__version__)

    # global count
    # global -> error here
    # count = 0
    setParam()

    imgInst = ImgClass()
    calibInst = CalibClass()
    dataLoadInst = DataLoadClass(imgInst, calibInst)

    # global flag_1_subscribeImg_2_loadImgFile_3_image2video
    rospy.init_node('carView', anonymous=True)
    try:
        if flag_1_subscribeImg_2_loadImgFile_3_image2video == 1:
            # One python file for one init_node
            if flag_publishImg == 1:
                print('flag_publishImg == ', flag_publishImg)
                dataLoadInst.publishImg()
                dataLoadInst.serviceCarInfo()

            dataLoadInst.subscribeImg()
            rospy.spin()

        elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 2:
            if flag_publishImg == 1:
                print('flag_publishImg == ', flag_publishImg)
                dataLoadInst.publishImg()
                dataLoadInst.serviceCarInfo()

            dataLoadInst.loadImgInFolder()



        elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 3:
            dataLoadInst.image2video()

    except KeyboardInterrupt:
        print("Shutting down")
