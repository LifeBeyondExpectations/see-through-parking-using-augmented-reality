#!/usr/bin/env python
from __future__ import division

import numpy as np
import cv2
from KalmanFilter import *
from math import pi
from random import randint


class LineSegClass:
    flag_imshow_on = 0
    flag_print_on = 0
    flag_first_init_kalman = 1
    end_of_arrow_pixel = None
    theta = 0
    kernelSize_close = (50, 50)
    kernelSize_open = (30, 30)
    boxColor = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0))
    kalmanInst = None

    def __init__(self, flag_imshow_on, flag_print_on, cctvLocation3D=None, carLocationDepth=None):
        self.flag_imshow_on = flag_imshow_on
        self.flag_print_on = flag_print_on

        if cctvLocation3D is not None:
            self.cctvLocation3D = cctvLocation3D
            self.carLocationDepth = carLocationDepth #height of the car from the ground. I just used 'depth' to clarify the width-height term.
        else:
            self.cctvLocation3D = None
            self.carLocationDepth = None

        global flag_1_measureONLY_2_kalman

    def redLineSegmentation(self, frame, pixelLeftTop=None, pixelRightBottom=None):

        if self.flag_imshow_on == 2:
            cv2.namedWindow('origin figure')
            cv2.imshow('origin figure', frame)
            cv2.waitKey(1)

        # This is to partially segment the car using Kalman filter output.
        # because there are some colorful things so that globally colarized segmentation goes wrong.
        if (pixelLeftTop is None) and (self.kalmanInst is not None):
            pixelLeftTop = [frame.shape[1] - 1, frame.shape[0] - 1]
            pixelRightBottom = [0, 0]

            for corner in range(4):
                for width_or_height in range(2):
                    if self.kalmanInst[corner][width_or_height].x_pri[0][0] < pixelLeftTop[width_or_height]:
                        pixelLeftTop[width_or_height] = int(self.kalmanInst[corner][width_or_height].x_pri[0][0])

                    if self.kalmanInst[corner][width_or_height].x_pri[0][0] > pixelRightBottom[width_or_height]:
                        pixelRightBottom[width_or_height] = int(self.kalmanInst[corner][width_or_height].x_pri[0][0])

            #         print('corner', corner, 'width_or_height', width_or_height)
            #         print('kalmanInst')
            #         print(self.kalmanInst[corner][width_or_height].x_pri)
            # print('pixelLeftTop', pixelLeftTop)
            # print('pixelRightBottom', pixelRightBottom)

            #enlarge the roi
            for width_or_height in range(2):
                pixelLeftTop[width_or_height] = int(pixelLeftTop[width_or_height] - 20)
                pixelRightBottom[width_or_height] = int(pixelRightBottom[width_or_height] + 20)

            if pixelLeftTop[0] < 0:
                pixelLeftTop[0] = 0
            elif pixelLeftTop[1] < 0:
                pixelLeftTop[1] = 0

            if pixelRightBottom[0] >= frame.shape[1]:
                pixelRightBottom[0] = frame.shape[1]
            elif pixelRightBottom[1] >= frame.shape[0]:
                pixelRightBottom[1] = frame.shape[0]

        img_saturation = np.zeros(frame.shape[:2], dtype=np.uint8)
        if pixelLeftTop is not None:
            img_saturation[pixelLeftTop[1]:pixelRightBottom[1], pixelLeftTop[0]:pixelRightBottom[0]] = \
                cv2.cvtColor(frame[pixelLeftTop[1]:pixelRightBottom[1], pixelLeftTop[0]:pixelRightBottom[0], :], cv2.COLOR_BGR2HSV)[:, :, 1]
        else:
            img_saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]

        if self.flag_imshow_on == 2:
            print('img_saturation[0, 0] = ', img_saturation[0, 0])
            cv2.namedWindow('saturation_figure')
            cv2.imshow('saturation_figure', img_saturation)
            cv2.waitKey(1)



        # # threshold

        # # _, img_threshold_otsu = cv2.threshold(np.array(img_close, dtype=np.uint8), 80, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # imb_blur = cv2.GaussianBlur(img_saturation, (5, 5), 0)
        # _, img_threshold = cv2.threshold(np.array(imb_blur, dtype=np.uint8), 110, 255,
        #                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # _, img_threshold = cv2.threshold(img_saturation, 110, 255, cv2.THRESH_BINARY)  ############img_saturation##################################################img_close##########
        # if self.flag_imshow_on == 2:
        #     cv2.namedWindow('cv2.THRESH_BINARY')
        #     cv2.imshow('cv2.THRESH_BINARY', img_threshold)
        #     cv2.waitKey(1)
        #
        #
        # # start to closing the image(morpoholgy)
        # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernelSize_close)
        # img_close = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel_close)
        # if self.flag_imshow_on == 2:
        #     cv2.namedWindow('img_close (30, 30)')
        #     cv2.imshow('img_close (30, 30)', img_close)
        #     cv2.waitKey(1)





        # start to closing the image(morpoholgy)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernelSize_close)
        img_close = cv2.morphologyEx(img_saturation, cv2.MORPH_CLOSE, kernel_close)
        if self.flag_imshow_on == 2:
            cv2.namedWindow('img_close (30, 30)')
            cv2.imshow('img_close (30, 30)', img_close)
            cv2.waitKey(1)

        # threshold
        # _, img_threshold_otsu = cv2.threshold(np.array(img_close, dtype=np.uint8), 80, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, img_threshold = cv2.threshold(img_close, 110, 255, cv2.THRESH_BINARY)########  150 ####img_saturation##################################################img_close##########
        if self.flag_imshow_on == 2:
            cv2.namedWindow('cv2.THRESH_BINARY')
            cv2.imshow('cv2.THRESH_BINARY', img_threshold)
            cv2.waitKey(1)

        # global kernelSize_open
        # self.kernelSize_open = (10, 10)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernelSize_open)
        img_open = cv2.morphologyEx(img_threshold,###### img_close img_threshold########################################################################################
                                    cv2.MORPH_OPEN, kernel_open)
        if self.flag_imshow_on == 2:
            cv2.namedWindow('img_open' + str(self.kernelSize_open))
            cv2.imshow('img_open' + str(self.kernelSize_open), img_open)
            cv2.waitKey(1)

        # cv2.imshow('redLineSeg', np.concatenate((np.concatenate((frame[:,:,0], img_threshold), axis=1), np.concatenate((img_close, img_open), axis=1)), axis=0))
        # # cv2.imshow('redLineSeg', np.concatenate((frame[:, :, 0], img_saturation, img_threshold, img_open), axis=1))
        # cv2.waitKey(1)
        # # cv2.destroyAllWindows()

        return img_open

    def adjustLocationError(self, carLocation2D):
        # result = carLocation2D
        if self.cctvLocation3D is not None:
            ratio = self.carLocationDepth / self.cctvLocation3D[2]
            for width_or_height in range(2):
                carLocation2D[width_or_height] = (ratio * self.cctvLocation3D[width_or_height]) + ((1.0 - ratio) * carLocation2D[width_or_height])

        # return carLocation2D


    def startDrawCircle(self, frame, img_open):

        # connected compnent labeling
        num_labels, img_connectedComponents, stat, centroid = cv2.connectedComponentsWithStats(image=img_open, connectivity=8, ltype=cv2.CV_16U)
        # print('num_labels are', num_labels)
        # print('stat is ', stat)
        # print('centroid is ', centroid)

        for label in range(1, num_labels):  # label == 0 is background
            cv2.circle(frame, tuple(np.array(centroid[label, :], np.uint16)), radius=10, color=self.boxColor[label], thickness=3)

        # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        # https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
        # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
        _, contours, _ = cv2.findContours(image=img_open, mode=1, method=2)  # , hierarchy
        ######### LIST out of index ERROR : might be no .. car detected ... ##################################################################

        # print('contours is ', contours)
        if contours != [] and contours is not None:
            cnt = contours[0]  ##   0 ##########################################################################3 fucking code ##################
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # # for comparison with measurementONLY
            # for i in range(box.shape[0]):
            #     cv2.circle(frame, tuple(box[i]), radius=5, color=tuple(self.boxColor[i]), thickness=2)

            if self.flag_first_init_kalman == 1:

                self.kalmanInst = [[KalmanFilter(QfactorProcessNoise = 0.001, Qfactor = 0.1), KalmanFilter(QfactorProcessNoise = 0.001, Qfactor = 0.1)] for i in range(4)]  # 4 : 4 corners at the car
                # print('shape of the self.kalmanInst is', np.shape(self.kalmanInst))
                # detail of index of 4by2 matrix
                # [0, 0] : self.kalmanInst_width_front_right = KalmanFilter()
                # [0, 1] : self.kalmanInst_height_front_right = KalmanFilter()
                # [1, 0] : self.kalmanInst_width_front_left = KalmanFilter()
                # [1, 1] : self.kalmanInst_height_front_left = KalmanFilter()
                # [2, 0] : self.kalmanInst_width_rear_right = KalmanFilter()
                # [2, 1] :self.kalmanInst_height_rear_right = KalmanFilter()
                # [3, 0] : self.kalmanInst_width_rear_left = KalmanFilter()
                # [3, 1] :self.kalmanInst_height_rear_left = KalmanFilter()

                # find the front of the car. sometimes box[] is not correctly correspondent to what I meant.
                boxClosestIndex = [0, 1, 2, 3]
                boxIndex_in_order_of_kalman_index = np.zeros(4, dtype=np.uint8)
                self.fineClosestCorner(box, boxClosestIndex, boxIndex_in_order_of_kalman_index)

                for corner in boxIndex_in_order_of_kalman_index:  # range(4):
                    for width_or_height in range(2):
                        self.kalmanInst[corner][width_or_height].initialMeasurement(z=box[corner, width_or_height])

                    # draw circle
                    cornerPixel = tuple((self.kalmanInst[corner][0].x_post[0][0], self.kalmanInst[corner][1].x_post[0][0]))
                    # print('cornerPixel is ', cornerPixel)
                    cv2.circle(frame, cornerPixel, radius=5, color=self.boxColor[corner], thickness=2)

                # centroid kalman filter
                self.kalmanInstCentroid = [KalmanFilter(QfactorProcessNoise = 0.001, Qfactor = 0.1), KalmanFilter(QfactorProcessNoise = 0.001, Qfactor = 0.1)]
                for width_or_height in range(2):
                    self.kalmanInstCentroid[width_or_height].initialMeasurement(z=centroid[1, width_or_height])  # 3  ######################################### should change '3' .. this is just for indicating that label(==3 is the car)

                # arrow kalman filter
                self.kalmanInstEndPixelOfArrow = [KalmanFilter(QfactorProcessNoise = 0.001, Qfactor = 0.1), KalmanFilter(QfactorProcessNoise = 0.001, Qfactor = 0.1)]

                # we assume that the end_of_arrow_pixel has smaller height value than the height value of the centroid.
                # this is because we know where the car start in the project.
                self.which_corner_defines_end_of_arrow_pixel = 0  # [label00, label01]
                self.end_of_arrow_pixel = np.zeros(2, dtype=np.int32)  # [[label00_width, label00_height], [label01_width, label01_height]]

                # car go straight at the first time in cctvView
                for corner in range(4):
                    if self.kalmanInst[corner][1].x_post[0][0] < centroid[1, 1]:  # 3  ######################################### should change '3' .. this is just for indicating that label(==3 is the car)
                        self.which_corner_defines_end_of_arrow_pixel = corner
                        break

                # print('self.which .. == ', self.which_corner_defines_end_of_arrow_pixel)
                for width_or_height in range(2):
                    self.end_of_arrow_pixel[width_or_height] = int((self.kalmanInst[self.which_corner_defines_end_of_arrow_pixel][width_or_height].x_post[0][0]
                                                                    + self.kalmanInst[boxClosestIndex[self.which_corner_defines_end_of_arrow_pixel]][width_or_height].x_post[0][0]) // 2)

                    # origin_of_arrow_pixel is centroid of the car
                    self.kalmanInstEndPixelOfArrow[width_or_height].initialMeasurement(z=self.end_of_arrow_pixel[width_or_height])

                # # draw arrow
                # origin_of_arrow = tuple(np.stack((self.kalmanInstCentroid[0].x_post[0][0], self.kalmanInstCentroid[1].x_post[0][0])))
                # end_of_arrow_pixel = tuple(np.stack((self.kalmanInstEndPixelOfArrow[0].x_post[0][0], self.kalmanInstEndPixelOfArrow[1].x_post[0][0])))
                # cv2.arrowedLine(img=frame, pt1=origin_of_arrow, pt2=end_of_arrow_pixel, color=(255, 255, 50), thickness=3)  # , line_type=None, shift=None, tipLength=None

                # do not come here again
                self.flag_first_init_kalman = 0

            ## second iterative kalman filter
            else:
                for corner in range(4):
                    # print('for ', corner, '-th corner')
                    cv2.circle(frame, tuple(box[corner]), radius=10, color=(0, 0, 255), thickness=2)

                    for width_or_height in range(2):
                        # print('width_or_height is ', width_or_height)

                        # box[width, height] is the corner pixel at the rectangle
                        min = np.sqrt(np.power((self.kalmanInst[corner][0].z_prior - box[0, 0]), 2) + np.power((self.kalmanInst[corner][1].z_prior - box[0, 1]), 2))
                        minIndex = 0

                        # print('np.shape(box)=', np.shape(box))
                        for label in range(1, box.shape[0]):
                            tmp = np.sqrt(np.power((self.kalmanInst[corner][0].z_prior - box[label, 0]), 2)
                                          + np.power((self.kalmanInst[corner][1].z_prior - box[label, 1]), 2))
                            if min > tmp:
                                min = tmp
                                minIndex = label

                        # print('min and minIndex is', min, minIndex)
                        self.kalmanInst[corner][width_or_height].measurementUpdate(z=np.array((box[minIndex, width_or_height]), dtype=np.float32))

                    # draw circle
                    cornerPixel = tuple((self.kalmanInst[corner][0].x_post[0][0], self.kalmanInst[corner][1].x_post[0][0]))
                    cv2.circle(frame, cornerPixel, radius=5, color=self.boxColor[corner], thickness=2)

                    # adjust the projection error for the carLocation. error cames from the assumption that the depth(height) of the car is zero.
                    # beforeBox = np.array(box[minIndex])
                    # self.adjustLocationError(carLocation2D=box[minIndex])
                    # print('adjustment value is ', box[minIndex] - beforeBox)
                    # cv2.circle(frame, tuple(box[corner]), radius=5, color=(0,255,0), thickness=2)




                # assume box have their pixel location along the edges
                box_middle_pixel = np.zeros(np.shape(box), dtype=np.int32)
                label_find_closest_box_middle_with_prior_end_of_arrow = -1
                min = 99999999
                for corner in range(4):
                    for width_or_height in range(2):
                        # box_middle_pixel[corner][width_or_height] = int((box[corner][width_or_height] + box[boxIndex_in_order_of_kalman_index[corner-1]][width_or_height]) // 2)
                        box_middle_pixel[corner][width_or_height] = int((box[corner][width_or_height] + box[corner - 1][width_or_height]) // 2)

                    # draw box_middle_pixel
                    # cv2.circle(frame, tuple(box_middle_pixel[corner]), radius=5, color=self.boxColor[corner], thickness=2)
                    if min > (np.power(self.kalmanInstEndPixelOfArrow[0].x_post[0][0] - box_middle_pixel[corner][0], 2) + np.power(self.kalmanInstEndPixelOfArrow[1].x_post[0][0] - box_middle_pixel[corner][1], 2)):
                        min = (np.power(self.kalmanInstEndPixelOfArrow[0].x_post[0][0] - box_middle_pixel[corner][0], 2) + np.power(self.kalmanInstEndPixelOfArrow[1].x_post[0][0] - box_middle_pixel[corner][1], 2))
                        label_find_closest_box_middle_with_prior_end_of_arrow = corner

                for width_or_height in range(2):
                    self.kalmanInstEndPixelOfArrow[width_or_height].measurementUpdate(z=box_middle_pixel[label_find_closest_box_middle_with_prior_end_of_arrow][width_or_height])

                # centroid kalman filter,
                for width_or_height in range(2):
                    sum = 0
                    for corner in range(4):
                        sum = sum + self.kalmanInst[corner][width_or_height].x_post[0][0]
                    self.kalmanInstCentroid[width_or_height].measurementUpdate(z=sum // 4)

            # draw arrow
            origin_of_arrow = tuple(np.stack((self.kalmanInstCentroid[0].x_post[0][0], self.kalmanInstCentroid[1].x_post[0][0])))
            # origin_of_arrow = tuple(np.concatenate((self.kalmanInstCentroid[0].x_post[0][0], self.kalmanInstCentroid[1].x_post[0][0])))
            # self.end_of_arrow = tuple(np.concatenate((self.kalmanInstEndPixelOfArrow[0].x_post[0][0], self.kalmanInstEndPixelOfArrow[1].x_post[0][0])))
            self.end_of_arrow = tuple(np.stack((int(self.kalmanInstEndPixelOfArrow[0].x_post[0][0]), int(self.kalmanInstEndPixelOfArrow[1].x_post[0][0]))))
            cv2.arrowedLine(img=frame, pt1=origin_of_arrow, pt2=self.end_of_arrow, color=(255, 255, 50), thickness=3)  # , line_type=None, shift=None, tipLength=None

            self.theta = self.getTheta(origin_of_arrow)
            cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 50), thickness=3)

            # I want to send self.theta with type 'int'
            # self.theta = int(self.theta)

            if self.flag_imshow_on == 2:
                # print('result of the redLineSeg is ', self.end_of_arrow, self.theta)
                cv2.namedWindow('redLineSeg')
                # cv2.imshow('redLineSeg', np.concatenate((frame[:, :, 1], img_open), axis=1))
                cv2.imshow('redLineSeg', frame)
                cv2.waitKey(0)

            self.adjustLocationError(self.end_of_arrow)
        return frame, self.end_of_arrow, self.theta

    def fineClosestCorner(self, box, boxClosestIndex, boxIndex_in_order_of_kalman_index):
        boxIndex = [0, 1, 2, 3]
        for corner in boxIndex:
            # print('corner=',corner)
            dist_corner_to_corner_min = 9999999
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
        if boxIndex_in_order_of_kalman_index[1] == 1:  # corner == 1 is next to the corner == 0
            if (np.power(box[1][0] - box[2][0], 2) + np.power(box[1][1] - box[2][1], 2)) < (np.power(box[1][0] - box[3][0], 2) + np.power(box[1][1] - box[3][1], 2)):
                boxIndex_in_order_of_kalman_index[2] = 2
                boxIndex_in_order_of_kalman_index[3] = 3
            else:
                boxIndex_in_order_of_kalman_index[2] = 3
                boxIndex_in_order_of_kalman_index[3] = 2
        elif boxIndex_in_order_of_kalman_index[1] == 3:  # corner == 3 is next to the corner == 0
            if (np.power(box[3][0] - box[1][0], 2) + np.power(box[3][1] - box[1][1], 2)) < (np.power(box[3][0] - box[2][0], 2) + np.power(box[3][1] - box[2][1], 2)):
                boxIndex_in_order_of_kalman_index[2] = 1
                boxIndex_in_order_of_kalman_index[3] = 2
            else:
                boxIndex_in_order_of_kalman_index[2] = 2
                boxIndex_in_order_of_kalman_index[3] = 1
        elif boxIndex_in_order_of_kalman_index[1] == 2:  # corner == 2 is next to the corner == 0
            if (np.power(box[2][0] - box[1][0], 2) + np.power(box[2][1] - box[1][1], 2)) < (np.power(box[2][0] - box[3][0], 2) + np.power(box[2][1] - box[3][1], 2)):
                boxIndex_in_order_of_kalman_index[2] = 1
                boxIndex_in_order_of_kalman_index[3] = 3
            else:
                boxIndex_in_order_of_kalman_index[2] = 3
                boxIndex_in_order_of_kalman_index[3] = 1

    def getTheta(self, origin_of_arrow):
        # calculate the theta
        horizontal_length = (self.end_of_arrow[0] - origin_of_arrow[0])
        vertical_length = (self.end_of_arrow[1] - origin_of_arrow[1])
        if horizontal_length >= 0 and vertical_length >= 0:  # range : 0 ~ pi/2
            theta = np.arctan(horizontal_length / vertical_length) + pi
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.boxColor[0], thickness=3)

        elif horizontal_length >= 0 and vertical_length < 0:  # range : pi/2 ~ pi
            theta = np.arctan(horizontal_length / vertical_length) + (2 * pi)
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.boxColor[1], thickness=3)

        elif horizontal_length < 0 and vertical_length < 0:
            theta = np.arctan(horizontal_length / vertical_length)
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.boxColor[2], thickness=3)

        elif horizontal_length < 0 and vertical_length >= 0:  # - pi/2 ~ 0
            theta = np.arctan(horizontal_length / vertical_length) + pi
            # self.theta = (self.theta * 180 / pi)
            # cv2.putText(img=frame, text=str(self.theta), org=self.end_of_arrow, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.boxColor[3], thickness=3)

        # print(' >> self.theta, horizontal_length, vertical_length is  ', self.theta, horizontal_length, vertical_length)
        return (theta * 180 / pi)

    def emptyParkingLotDetection(self, frame, parkingLotCenter):
        img_saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        if self.flag_imshow_on == 2:
            cv2.imshow('img_saturation', img_saturation)
            cv2.waitKey(1)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 80)) ###################################################################
        img_close = cv2.morphologyEx(img_saturation, cv2.MORPH_CLOSE, kernel_close)
        if self.flag_imshow_on == 2:
            cv2.imshow('img_close', img_close)
            cv2.waitKey(1)

        _, img_threshold = cv2.threshold(img_close, thresh=70, maxval=255, type=cv2.THRESH_BINARY_INV) ##########################################################
        if self.flag_imshow_on == 2:
            cv2.imshow('img_thresh', img_threshold)
            cv2.waitKey(1)

        open_parking_lot = []
        for parkingLot in range(np.shape(parkingLotCenter)[0]):
            sum = 0
            sum = np.sum(img_threshold[parkingLotCenter[parkingLot][1] : (parkingLotCenter[parkingLot][1] + 5), parkingLotCenter[parkingLot][0] : (parkingLotCenter[parkingLot][0] + 5)])
            # print('np.sum(..)=', sum)
            if sum > 6000:
                open_parking_lot.append(parkingLot)

        print('open_parking_lot is ', open_parking_lot)




        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        # img_open = cv2.morphologyEx(img_threshold, cv2.MORPH_DILATE, kernel)
        # if self.flag_imshow_on == 1:
        #     cv2.imshow('img_open', img_open)
        #     cv2.waitKey(1)


        # _, img_threshold = cv2.threshold(img_saturation, thresh=110, maxval=255, type=cv2.THRESH_BINARY)
        # if self.flag_imshow_on == 1:
        #     cv2.imshow('img_thresh', img_threshold)
        #     cv2.waitKey(1)

        # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernelSize_close)
        # img_close = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel_close)
        # if self.flag_imshow_on == 1:
        #     cv2.imshow('img_close', img_close)
        #     cv2.waitKey(1)

        # I am not sure about the python compiler
        # img_subtract = np.zeros(img_close.shape, dtype=np.uint8)
        # img_subtract = img_threshold - img_close
        # if self.flag_imshow_on == 2:
        #     cv2.imshow('img_subract', img_subtract)
        #     cv2.waitKey(0)

        # cv2.imshow('emptyParkingLot', np.concatenate((img_close, img_threshold), axis=1)) #img_saturation, , img_subtract , img_open
        # cv2.waitKey(1)


        # # connected compnent labeling
        # # num_labels, img_connectedComponents = cv2.connectedComponents(image=img_threshold, connectivity=8)
        # num_labels, img_connectedComponents, stat, centroid = cv2.connectedComponentsWithStats(image=img_threshold, # img_open,  #
        #                                                                                        connectivity=8, ltype=cv2.CV_16U)
        #
        # color = [[randint(180, 255), randint(180, 255), randint(180, 255)] for i in range(num_labels)]
        # img_label = np.zeros(frame.shape, dtype=np.uint8)
        # for i in range(img_label.shape[0]):
        #     for j in range(img_label.shape[1]):
        #         if img_connectedComponents[i, j] > 1:  # for BINARY_THRESHOLD_INV, label == 0 and label == 1 means background
        #             # the reason " > 1 " is that we do know want background to be labeled.
        #             img_label[i, j] = color[img_connectedComponents[i, j]]
        #         else: #if img_whiteLine[i, j] == 255:
        #             img_label[i, j] = np.array([0, 255, 255], dtype=np.uint8)
        #
        #
        #
        # # using number of CCL. old fashion
        # # print('stat', stat)
        # # open_parking_lot = []
        # # for i in range(2, num_labels): # for BINARY_THRESHOLD_INV, label == 0 and label == 1 means background
        # #     if stat[i][4] > 6000 :
        # #         cv2.putText(img=img_label, text=str(i), org=(int(centroid[i][0]), int(centroid[i][1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=5)
        # #         open_parking_lot.append(i)
        # # print('open_parking_lot is ', open_parking_lot)
        #
        #
        #
        #
        # for parkingLot in open_parking_lot:
        #     cv2.putText(img=img_label, text=str(parkingLot), org=(parkingLotCenter[parkingLot][0], parkingLotCenter[parkingLot][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), thickness=5)
        #
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_label')
        #     cv2.imshow('img_label', np.concatenate((np.stack((img_threshold,
        #                                                       img_threshold,
        #                                                       img_threshold), axis=2),
        #                                             img_label,
        #                                             frame), axis=1))
        #     cv2.waitKey(1)
        #
        # # # the reason is that we do know want background, red lines, non-parking area(label = 6) to be labeled.
        # # num_labels = num_labels - 3
        # #
        # # if self.flag_print_on == 1:
        # #     print('what is the result', np.shape(img_connectedComponents), 'max is ',
        # #           np.max(img_connectedComponents))
        # #     print('num_labels is ', num_labels)
        # #
        # # return img_label


    #
    # def whiteLineSegmentation(self, frame):
    #
    #     # print('shape of the frame is ', np.shape(frame))
    #     frame = cv2.resize(frame, (1062, 598))
    #
    #     if self.flag_imshow_on == 1:
    #         cv2.namedWindow('origin_figure')
    #         cv2.imshow('origin_figure', frame)
    #         cv2.waitKey(0)
    #
    #     # convert rgb to hsv
    #     img_hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    #
    #     if self.flag_imshow_on == 1:
    #         cv2.namedWindow('hue figure')
    #         cv2.imshow('hue figure', img_hue)
    #         cv2.waitKey(0)
    #
    #     # start to opening the image(morpoholgy)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    #     # np.zeros((300, img_hue.shape[1]), dtype=np.uint8)
    #     img_open = cv2.morphologyEx(img_hue, cv2.MORPH_OPEN, kernel)
    #     print('dtype check', img_open.dtype)
    #     if self.flag_imshow_on == 1:
    #         cv2.namedWindow('img_open')
    #         cv2.imshow('img_open', img_open)
    #         print('max of img_open is ', np.max(img_open))
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #
    #     img_sub = np.zeros(img_hue.shape, dtype=np.uint8)
    #     for i in range(0, img_hue.shape[0]):
    #         for j in range(0, img_hue.shape[1]):
    #             if img_hue[i, j] < img_open[i, j]:
    #                 img_sub[i, j] = 0
    #             else:
    #                 img_sub[i, j] = img_hue[i, j] - img_open[i, j]
    #
    #     if self.flag_imshow_on == 1:
    #         cv2.namedWindow('img_sub')
    #         cv2.imshow('img_sub', img_sub)
    #         cv2.waitKey(0)
    #     if self.flag_print_on == 2:
    #         print('shape of the img_sub is ', img_sub.shape)
    #
    #     # start to closing the image(morpoholgy)
    #     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    #     img_close = cv2.morphologyEx(img_sub, cv2.MORPH_OPEN, kernel_close)
    #     if self.flag_imshow_on == 1:
    #         cv2.namedWindow('img_close')
    #         cv2.imshow('img_close', img_close)
    #         cv2.waitKey(0)
    #
    #     img_sub = img_close
    #     # #column wise threshold
    #     # img_col_wise_otsu_threshold = np.zeros(img_sub.shape, dtype=np.uint8)
    #     # img_col_wise_adoptive_threshold = np.zeros(img_sub.shape, dtype=np.uint8)
    #     #
    #     # for col in range(img_col_wise_otsu_threshold.shape[1]):
    #     #
    #     #     _, tmp = cv2.threshold(np.array(img_sub[:, col], dtype=np.uint8), 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     #     img_col_wise_otsu_threshold[:, col] = tmp.reshape(1, img_sub.shape[0])
    #     #
    #     #     img_col_wise_adoptive_threshold = cv2.adaptiveThreshold(np.array(img_sub, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
    #     #
    #     # if self.flag_imshow_on == 1:
    #     #     cv2.imshow('img_row_wise_otsu_threshold', np.concatenate((img_col_wise_otsu_threshold, img_col_wise_adoptive_threshold), axis=1))
    #     #     cv2.waitKey(0)
    #     #
    #     #     # start to opening the image(morpoholgy)
    #     #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    #     #     # np.zeros((300, img_hue.shape[1]), dtype=np.uint8)
    #     #     img_col_wise_adoptive_threshold = cv2.morphologyEx(img_col_wise_adoptive_threshold, cv2.MORPH_OPEN, kernel)
    #     #
    #     #     if self.flag_imshow_on == 1:
    #     #         cv2.namedWindow('img_col_wise_adoptive_threshold_opened')
    #     #         cv2.imshow('img_col_wise_adoptive_threshold_opened', img_col_wise_adoptive_threshold)
    #     #         cv2.waitKey(0)
    #
    #     # original
    #     _, img_threshold = cv2.threshold(img_sub, 40, 255, cv2.THRESH_BINARY)
    #
    #     _, img_threshold_otsu = cv2.threshold(np.array(img_sub, dtype=np.uint8), 10, 255,
    #                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    #     imb_blur = cv2.GaussianBlur(img_sub, (5, 5), 0)
    #     _, img_threshold_otsu_blur = cv2.threshold(np.array(imb_blur, dtype=np.uint8), 30, 255,
    #                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    #     # # split -> otsh -> merge
    #     # middle_row = (img_sub.shape[0] // 2)
    #     # middle_col = (img_sub.shape[1] // 2)
    #     #
    #     # if self.flag_imshow_on == 1:
    #     #     cv2.namedWindow('img_sub_otsu_1')
    #     #     cv2.imshow('img_sub_otsu_1', img_sub)
    #     #     cv2.waitKey(0)
    #     #
    #     # _, img_split_otsu = cv2.threshold(np.array(img_sub[:middle_row, :middle_col], dtype=np.uint8), 10, 255,
    #     #                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     # img_sub[:middle_row, :middle_col] = img_split_otsu
    #     #
    #     # if self.flag_imshow_on == 1:
    #     #     cv2.namedWindow('img_sub_otsu_2')
    #     #     cv2.imshow('img_sub_otsu_2', img_sub)
    #     #     cv2.waitKey(0)
    #     #
    #     # _, img_split_otsu = cv2.threshold(np.array(img_sub[middle_row:, :middle_col], dtype=np.uint8), 10, 255,
    #     #                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     # img_sub[middle_row:, :middle_col] = img_split_otsu
    #     #
    #     # if self.flag_imshow_on == 1:
    #     #     cv2.namedWindow('img_sub_otsu_3')
    #     #     cv2.imshow('img_sub_otsu_3', img_sub)
    #     #     cv2.waitKey(0)
    #     #
    #     # _, img_split_otsu = cv2.threshold(np.array(img_sub[:middle_row, middle_col:], dtype=np.uint8), 10, 255,
    #     #                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     # img_sub[:middle_row, middle_col:] = img_split_otsu
    #     #
    #     # _, img_split_otsu = cv2.threshold(np.array(img_sub[middle_row:, middle_col:], dtype=np.uint8), 10, 255,
    #     #                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     # img_sub[middle_row:, middle_col:] = img_split_otsu
    #     # if self.flag_imshow_on == 1:
    #     #     cv2.namedWindow('img_sub_otsu')
    #     #     cv2.imshow('img_sub_otsu', img_sub)
    #     #     cv2.waitKey(0)
    #
    #
    #
    #
    #
    #
    #     # img_adaptive_threshold_mean = cv2.adaptiveThreshold(np.array(img_sub, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #     #                                                     cv2.THRESH_BINARY_INV, 11 ,2)
    #     # img_adaptive_threshold_gaussian = cv2.adaptiveThreshold(np.array(img_sub, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     #                                                cv2.THRESH_BINARY_INV, 19, 2)
    #
    #     # fucking code
    #     img_threshold_otsu[496:, :] = np.zeros((img_threshold_otsu.shape[0] - 496, img_threshold_otsu.shape[1]), dtype=np.uint8)
    #     img_threshold_otsu[317:440, 685:885] = np.zeros((123, 200), dtype=np.uint8)
    #     img_threshold_otsu[358:388, 896:934] = np.zeros((388 - 358, 934 - 896), dtype=np.uint8)
    #
    #     img_whiteLine = np.zeros(img_threshold_otsu.shape, dtype=np.uint8)
    #
    #     for i in range(img_threshold_otsu.shape[0]):
    #         for j in range(img_threshold_otsu.shape[1]):
    #
    #             # copy the white line area
    #             img_whiteLine[i, j] = img_threshold_otsu[i, j]
    #
    #             # Thresh_INV
    #             if img_threshold_otsu[i, j] == 0:
    #                 img_threshold_otsu[i, j] = 255
    #             else:
    #                 img_threshold_otsu[i, j] = 0
    #
    #     if self.flag_imshow_on == 1:
    #         # cv2.namedWindow('img_threshold')
    #         # cv2.imshow('img_threshold',
    #         #            np.concatenate((img_threshold, img_threshold_otsu, img_threshold_otsu_blur), axis=1))
    #         # img_adaptive_threshold_mean, img_adaptive_threshold_gaussian
    #         cv2.namedWindow('img_threshold_otsu')
    #         cv2.imshow('img_threshold_otsu', img_threshold_otsu)
    #         cv2.waitKey(0)
    #
    #     # connected compnent labeling
    #     num_labels, img_connectedComponents = cv2.connectedComponents(image=img_threshold_otsu,
    #                                                                   connectivity=8)
    #     color = np.zeros((num_labels, 3), dtype=np.uint8)
    #     I_need_only_one_color = [180, 180, 0]  # [randint(0, 255), randint(0, 255), randint(0, 255)]
    #     for i in range(num_labels):
    #         color[i] = I_need_only_one_color
    #
    #     img_label = np.zeros(frame.shape, dtype=np.uint8)
    #     for i in range(img_label.shape[0]):
    #         for j in range(img_label.shape[1]):
    #             if img_connectedComponents[i, j] > 1:  # and img_connectedComponents[i, j] != 5:
    #                 # the reason " > 1 " is that we do know want background to be labeled.
    #                 img_label[i, j] = color[img_connectedComponents[i, j]]
    #             elif img_whiteLine[i, j] == 255:
    #                 img_label[i, j] = np.array([0, 255, 255], dtype=np.uint8)
    #
    #     if self.flag_imshow_on == 1:
    #         cv2.namedWindow('img_label')
    #         cv2.imshow('img_label', img_label)
    #         cv2.waitKey(0)
    #
    #     # the reason is that we do know want background, red lines, non-parking area(label = 6) to be labeled.
    #     num_labels = num_labels - 3
    #
    #     if self.flag_print_on == 1:
    #         print('what is the result', np.shape(img_connectedComponents), 'max is ',
    #               np.max(img_connectedComponents))
    #         print('num_labels is ', num_labels)
    #
    #     return img_label  # img_threshold_otsu