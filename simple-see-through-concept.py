#!/usr/bin/env python

from __future__ import print_function
import pprint

import numpy as np
from random import randint
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import rospy
import roslib

import pickle
from tempfile import TemporaryFile

import glob
import os

#parameter
path_src_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/background_origin.jpg'
path_image_database = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/background_origin.jpg'
save_video_to_the_path = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/fisheye-calibration-opencv-python-ROS/staticEndVersion/171116/undistort/roi/'
save_concat_result = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/simpleVersion/result/'

#flag
flag_set_imshow_on = 1
flag_set_print_on = 1
#flag_hand_homography_on = 0
flag_1_warp_entire_src_2_mask_src = 2
flag_set_white_lane_segmentation = 1
flag_is_compressed_image = 1
where_to_start = 10000

class calibration:
    # you must check the checkboard which you would like to use for the calibration
    # if (9, 7) it means there are 10(=9+1) pactches in row, 8(=7+1) patches in column
    CHECKERBOARD = (9, 7)  # (8, 8)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = []

    flag_calibratedResult_save = 1
    flag_get_undistort_param = 1
    flag_first_didHomography = 1

    height_train = 0
    width_train = 0

    height_test = 0
    width_test = 0

    def __init__(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

    def detectCheckerBoard(self, frame):
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # https://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
        ret, corners = cv2.findChessboardCorners(image=gray,
                                                 patternSize=self.CHECKERBOARD,
                                                 flags=(cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE))

        # If found, add object points, image points (after refining them)
        if ret == True:
            print(">> detected the checkerboard")

            # Save images if wanted
            if flag_subscribe_new_image_not_load_old_image == 1 and flag_save_image_onlyWhichDetectCheckeboard == 1:
                cv2.imwrite(save_video_to_the_path + (str((count + 1000)) + '.png'), frame)

            # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#void cornerSubPix(InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria criteria)
            corners2 = cv2.cornerSubPix(image=gray,
                                        corners=corners,
                                        winSize=(11, 11),
                                        zeroZone=(-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            self.imgpoints.append(corners2)
            self.objpoints.append(self.objp)

            # Draw and display the corners
            frame = cv2.drawChessboardCorners(image=frame, patternSize=self.CHECKERBOARD, corners=corners2, patternWasFound=ret)

        # Display the resulting frame
        cv2.namedWindow('frame')
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    def saveVarInDetection(self):
        with open(nameOf_pickle_Checkerboard_Detection, 'w') as f:
            pickle.dump([self.objpoints, self.imgpoints, self.width_train, self.height_train], f)

    def loadVarInDetection(self):
        with open(nameOf_pickle_Checkerboard_Detection) as f:
            self.objpoints, self.imgpoints, self.width_train, self.height_train = pickle.load(f)

            # check the result
            tmp = np.array(self.objpoints)
            print("shape of objpoints is ", tmp.shape)

            tmp = np.array(self.imgpoints)
            print("shape of imgpoints is ", tmp.shape)

            print("loadVarInDetection complete', ' width_train is ", self.width_train, "height_train is ", self.height_train)

    def startCalibration(self):
        global mybalance

        if flag_fisheye_calibrate == 0:

            N_OK = len(self.objpoints)
            self.camera_matrix = np.zeros((3, 3))
            self.distCoeffs = np.zeros((4, 1))
            self.rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            self.tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]


            self.ret, _, _, _, _ = \
                cv2.calibrateCamera(
                objectPoints=self.objpoints, imagePoints=self.imgpoints, imageSize=(self.width, self.height),
                cameraMatrix=self.camera_matrix, distCoeffs=self.distCoeffs, rvecs=self.rvecs, tvecs=self.tvecs,
                flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

            # check the result of calibration
            print('RMS re-projection error is ', self.ret)
            print('distort Coeffs is ')
            print(self.distCoeffs)
            print('camera matrix is ')
            print(self.camera_matrix)

            # https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#getoptimalnewcameramatrix
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix,
                                                                             self.distCoeffs,
                                                                             (self.width, self.height), 1.0,
                                                                             (self.width, self.height))
            print("self.roi is ", self.roi)

            ## self.roi or (self.width, self.height) ??
            self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.distCoeffs, np.eye(3),
                                                               self.new_camera_matrix,
                                                               (self.width, self.height),
                                                               cv2.CV_16SC2)


        else:

            # self.ret, self.camera_matrix, self.distCoeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(self.objpoints, self.imgpoints, (self.width, self.height), None, None)
            # you should write all the cv2.fisheye.CALIB_..3 things .. then it works



            # # The result is same with originally works
            # self.ret, self.camera_matrix, self.distCoeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
            #     self.objpoints, self.imgpoints, (self.width, self.height), None, None, None, None,
            #     cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
            #     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

            # self.ret, self.camera_matrix, self.distCoeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
            #     self.objpoints, self.imgpoints, (self.width, self.height), None, None, None, None,
            #     cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            #     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

            # No good results at all
            # self.ret, self.camera_matrix, self.distCoeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
            #     self.objpoints, self.imgpoints, (self.width, self.height), None, None, None, None,
            #     cv2.fisheye.CALIB_CHECK_COND,
            #     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

            # No good results at all
            # self.ret, self.camera_matrix, self.distCoeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
            #     self.objpoints, self.imgpoints, (self.width, self.height), None, None, None, None,
            #     cv2.fisheye.CALIB_FIX_INTRINSIC,
            #     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

            # Does not work at all
            # self.ret, self.camera_matrix, self.distCoeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
            #     self.objpoints, self.imgpoints, (self.width, self.height), None, None)

            # originally works
            # self.ret, self.camera_matrix, self.distCoeffs, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
            #     self.objpoints, self.imgpoints, (self.width, self.height), None, None, None, None,
            #     cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
            #     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

            # originally works
            print('before calibartion, width_train, height_train is ', self.width_train, self.height_train)
            N_OK = len(self.objpoints)
            self.camera_matrix = np.zeros((3, 3))
            self.distCoeffs = np.zeros((4, 1))
            self.rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            self.tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            self.ret, _, _, _, _ = cv2.fisheye.calibrate(
                objectPoints=self.objpoints,
                imagePoints=self.imgpoints,
                image_size=(self.width_train, self.height_train),
                K=self.camera_matrix,
                D=self.distCoeffs,
                rvecs=None,  # self.rvecs,
                tvecs=None,  # self.tvecs,
                flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND), #  cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6))

        # check the result of calibration
        print('camera matrix is ')
        print(self.camera_matrix)
        # print('self.rvecs is ')
        # print(self.rvecs)
        # print('self.tvecs is ')
        # print(self.tvecs)
        print('distort Coeffs is ')
        print(self.distCoeffs)
        print('RMS re-projection error is ', self.ret)
        print('balance is ', mybalance)
        print("calibration complete")

    def saveVarAfterCalibration(self):

        if flag_fisheye_calibrate == 0:
            with open(nameof_pickel_calibrated_result, 'w') as f:
                pickle.dump([self.camera_matrix, self.distCoeffs, self.width_train, self.height_train], f)
            print('self.roi is', self.roi)
        else:
            with open(nameof_pickel_calibrated_result, 'w') as f:
                pickle.dump([self.camera_matrix, self.distCoeffs, self.width_train, self.height_train], f)

    def loadVarAfterCalibration(self):

        with open(nameof_pickel_calibrated_result) as f:
            self.camera_matrix, self.distCoeffs, self.width_train, self.height_train = pickle.load(f)
            #self.camera_matrix, self.distCoeffs, self.new_camera_matrix, self.width_train, self.height_train, self.roi, self.map1, self.map2  = pickle.load(f)
            #for old data, erase self.roi, self.map1, self.map2

        print('camera matrix is ')
        print(self.camera_matrix)
        print('distort Coeffs is ')
        print(self.distCoeffs)
        print('balance is ', mybalance, 'loadVarAfterCalibration complete')

    def undistort_imShow(self, frame):
        trainImageShape = (self.width_train, self.height_train)
        currentFrameShape = frame.shape[:2][::-1]
        dim3 = tuple((np.array(currentFrameShape) / 1).astype(int))
        displayDim = (640, 480)#tuple((np.array(currentFrameShape) / 1.3).astype(int))
        # print('trainImageShapeis ', trainImageShape, 'currentFrameShape is ', currentFrameShape, 'displayDim is ', displayDim)

        if flag_fisheye_calibrate == 0:

            # best
            # cv2.undistort : https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d
            # The function is simply a combination of initUndistortRectifyMap() (with unity R ) and remap() (with bilinear interpolation)
            # I cannot tell the difference between the two line below
            frame_with_new_camera_matrix = cv2.undistort(frame, self.camera_matrix, self.distCoeffs, None, newCameraMatrix=self.new_camera_matrix)
            frame_with_origin_camera_matrix = cv2.undistort(frame, self.camera_matrix, self.distCoeffs, None, None)

            # cv2.namedWindow('undistorted frame')
            #cv2.imshow('undistorted frame', frame_with_new_camera_matrix)

            frame_with_remap_origin = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.distCoeffs, np.eye(3),
                                                               self.new_camera_matrix,
                                                               (self.width, self.height),
                                                               cv2.CV_16SC2)


            # shit
            frame_with_remap = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            #cv2.imshow('undistorted frame', frame_with_remap)

            # compare with distorted / undistorted
            # cv2.imshow('undistorted frame', np.concatenate((frame_with_new_camera_matrix, frame), axis=1))
            # compare with camera_matrixes
            # cv2.imshow('undistorted frame', np.concatenate((frame_with_new_camera_matrix, frame_with_origin_camera_matrix), axis=1))
            # test
            cv2.imshow('undistorted frame', np.concatenate((cv2.resize(frame_with_new_camera_matrix, (self.width, self.height), cv2.INTER_LINEAR),
                                                            cv2.resize(frame_with_origin_camera_matrix, (self.width, self.height), cv2.INTER_LINEAR),
                                                            cv2.resize(frame_with_remap_origin, (self.width, self.height), cv2.INTER_LINEAR),
                                                            cv2.resize(frame_with_remap, (self.width, self.height), cv2.INTER_LINEAR),
                                                            cv2.resize(frame, (self.width, self.height), cv2.INTER_LINEAR)
                                                            ), axis=1))
            cv2.waitKey(1)

        elif flag_fisheye_calibrate == 1:

            if self.flag_get_undistort_param == 1:

                # self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K=self.camera_matrix,
                #                                                                                 D=self.distCoeffs,
                #                                                                                 image_size=trainImageShape,
                #                                                                                 R=np.eye(3),
                #                                                                                 #P=None,
                #                                                                                 balance=mybalance,
                #                                                                                 new_size=dim3#,fov_scale=1.0
                #                                                                                 )
                #
                # # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
                # self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K=self.camera_matrix,
                #                                                             D=self.distCoeffs,
                #                                                             R=np.eye(3),
                #                                                             P=self.camera_matrix,
                #                                                             size=currentFrameShape,
                #                                                             m1type=cv2.CV_32FC1)
                #
                # # check the result of calibration
                # print('camera matrix is ')
                # print(self.camera_matrix)
                # print('new camera Matrix is ')
                # print(self.new_camera_matrix)
                # print('distort Coeffs is ')
                # print(self.distCoeffs)

                self.flag_get_undistort_param = 0


            frame_undistorted_fisheye_camera_matrix = cv2.fisheye.undistortImage(frame, self.camera_matrix, self.distCoeffs, Knew=self.camera_matrix, new_size=currentFrameShape)
            # frame_undistorted_fisheye_remap_= cv2.remap(src=frame,map1=self.map1,map2=self.map2,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_DEFAULT)

            #tmp = cv2.resize(cv2.rotate(frame_undistorted_fisheye_camera_matrix, cv2.ROTATE_90_COUNTERCLOCKWISE), displayDim, cv2.INTER_LINEAR)
            tmp = cv2.resize(frame_undistorted_fisheye_camera_matrix, displayDim, cv2.INTER_LINEAR)
            global flag_show_image_2_homography_3_distorted
            if flag_show_image_2_homography_3_distorted == 1:
                # cv2.imshow('francoisLensFrame', np.concatenate((cv2.resize(frame_undistorted_fisheye_camera_matrix, displayDim, cv2.INTER_LINEAR),
                #                                          cv2.resize(frame_undistorted_fisheye_remap_, displayDim, cv2.INTER_LINEAR),
                #                                          cv2.resize(frame, displayDim, cv2.INTER_LINEAR)),axis=1))
                #tmp = cv2.resize(cv2.flip(frame_undistorted_fisheye_camera_matrix, flipCode=-1), displayDim, cv2.INTER_LINEAR)

                cv2.namedWindow('Francois calibrated frame')
                cv2.imshow('Francois calibrated frame', tmp)
            elif flag_show_image_2_homography_3_distorted == 2:
                tmp = self.wrapper_homography(tmp)
            elif flag_show_image_2_homography_3_distorted == 3:
                #tmp = frame
                cv2.namedWindow('Francois distorted frames')
                cv2.imshow('Francois distorted frames', frame)

            cv2.waitKey(1)
            return tmp
            # return cv2.resize(frame_undistorted_fisheye_camera_matrix, (10,10), cv2.INTER_LINEAR)
        else:
            print('error for <flag_fisheye_calibrate>')
            return None

    def homography(self, image_src=None, image_dst=None):
        # if self.flag_first_didHomography == 1:

            # top view
            # srcPoints = np.array([
            #     [269,102], [411, 144], [551, 128], [696,130],
            #     [265,241], [412,248], [546,249], [695, 248],
            #     [257,389], [420,384], [543,389], [694,384],
            #                [423,509], [550,512]
            # ])
            #
            # dstPoints = np.array([
            #     [189,294], [319,240], [427,203], [505,163],
            #     [282,355], [405,286], [500,234], [580,192],
            #     [387,428], [519,336], [600,277], [666,225],
            #                [629, 390], [705,329]
            # ])

        global count



        srcPoints, dstPoints = np.array(srcPoints), np.array(dstPoints)
        # print('dstPoints is ', dstPoints.shape, srcPoints.shape)

        self.homography_matrix, mask = cv2.findHomography(srcPoints=srcPoints, dstPoints=dstPoints, method=0)#RANSAC cv2.LMEDS
        # print('homography_matrix is ', self.homography_matrix)
        # self.homography_matrix = cv2.getPerspectiveTransform(srcPoints_Francois, dstPoints_Francois)
        # print('what is the shape of the image?' , image_src.shape)
        frame_homography_warpPerspective = cv2.warpPerspective(src=image_src, M=self.homography_matrix, dsize=image_dst.shape[:2][::-1], borderMode=cv2.BORDER_CONSTANT)

        #fuck
        frame_homography_warpPerspective[:526, :] = np.zeros((526, frame_homography_warpPerspective.shape[1], 3), dtype=np.uint8)

        # cv2.namedWindow('frame_homography_warpPerspective results')
        # cv2.imshow('frame_homography_warpPerspective results', frame_homography_warpPerspective)
        # cv2.waitKey(0)

        return frame_homography_warpPerspective

    def alphaBlending(self, image_src, image_dst, alpha):
        beta = (1 - alpha)

        image_blended = cv2.addWeighted(src1=image_src, alpha=alpha, src2=image_dst, beta=beta, gamma=0.0, dst=None, dtype=-1)
        # cv2.namedWindow(str('alphaBlended alpha = ' + str(alpha) + ' beta = ' + str(beta)))
        # cv2.imshow(str('alphaBlended alpha = ' + str(alpha) + ' beta = ' + str(beta)), image_blended)
        # cv2.waitKey(0)
        # print('image has been saved _', (save_video_to_the_path + str(count) + '.png'))
        # cv2.imwrite((save_video_to_the_path + str(count) + '.png'), image_blended)
        return image_blended

class JS_lineSegmentation:
    flag_imshow_on = 0
    flag_print_on = 0
    def __init__(self, flag_imshow_on, flag_print_on):
        self.flag_imshow_on = flag_imshow_on
        self.flag_print_on = flag_print_on

    def whiteLineSegmentation_new(self, frame):

        frame = cv2.resize(frame, (1062, 598))

        if self.flag_imshow_on == 1:
            cv2.namedWindow('origin_figure')
            cv2.imshow('origin_figure', frame)
            cv2.waitKey(0)

        # convert rgb to hsv
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        img_sat = img_hsv[:, :, 2]
        img_value = img_hsv[:, :, 1]

        _, img_sat_threshold = cv2.threshold(img_sat, 4, 255, cv2.THRESH_BINARY)
        if self.flag_imshow_on == 2:
            cv2.namedWindow('img_sat_threshold')
            cv2.imshow('img_sat_threshold', img_sat_threshold)
            cv2.waitKey(0)

        _, img_value_threshold = cv2.threshold(img_value, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.flag_imshow_on == 2:
            cv2.namedWindow('img_value_threshold')
            cv2.imshow('img_value_threshold', img_value_threshold)
            cv2.waitKey(0)

        img_my_method = self.whiteLineSegmentation_old(frame)

        img_and_operation = np.zeros(img_value_threshold.shape, dtype=np.uint8)
        for i in range(img_and_operation.shape[0]):
            for j in range(img_and_operation.shape[1]):
                if img_value_threshold[i, j] == 255 and img_sat_threshold[i, j] == 255 and img_my_method[i, j] == 255:
                    img_and_operation[i, j] = 255

        if self.flag_imshow_on == 2:
            cv2.imshow('img_and_operation', img_and_operation)
            cv2.waitKey(0)

        # start to opening the image(morpoholgy)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        img_open = cv2.morphologyEx(img_and_operation, cv2.MORPH_OPEN, kernel_open)
        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_open')
            cv2.imshow('img_open', img_open)
            cv2.waitKey(0)

    def whiteLineSegmentation_old_old(self, frame):

        if self.flag_imshow_on == 2:
            cv2.namedWindow('origin figure')
            cv2.imshow('origin figure', frame)
            cv2.waitKey(0)

        # convert rgb to hsv
        img_saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]

        if self.flag_imshow_on == 2:
            cv2.namedWindow('saturation_figure')
            cv2.imshow('saturation_figure', img_saturation)
            cv2.waitKey(0)

        # print('shape of the img_saturation is ', img_saturation.shape)
        img_row_wise_otsu_threshold = np.zeros(img_saturation.shape, dtype=np.uint8)
        for row in range(500, img_row_wise_otsu_threshold.shape[0]):
            _, tmp = cv2.threshold(img_saturation[row, :], 120, 255,
                            cv2.THRESH_BINARY)# + cv2.THRESH_OTSU
            img_row_wise_otsu_threshold[row, :] = tmp.reshape(1, img_saturation.shape[1])

        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_row_wise_otsu_threshold')
            cv2.imshow('img_row_wise_otsu_threshold', img_row_wise_otsu_threshold)
            cv2.waitKey(0)


        # # start to opening the image(morpoholgy)
        # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        # img_open = cv2.morphologyEx(img_saturation, cv2.MORPH_OPEN, kernel_open)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_open')
        #     cv2.imshow('img_open', img_open)
        #     cv2.waitKey(0)

        # # start to closing the image(morpoholgy)
        # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        # img_close = cv2.morphologyEx(img_saturation, cv2.MORPH_CLOSE, kernel_close)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_close')
        #     cv2.imshow('img_close', img_close)
        #     cv2.waitKey(0)

        # dilate image
        # for img in [img_saturation]: #img_threshold, img_threshold_otsu,
        # # blur makes small noise gone
        #     kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        #     img_dilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel_dilate)
        #     if self.flag_imshow_on == 1:
        #         cv2.namedWindow('img_dilate')
        #         cv2.imshow('img_dilate', img_dilate)
        #         cv2.waitKey(0)


        # threshold
        img_close = np.concatenate((np.zeros((300, img_saturation.shape[1]), dtype=np.uint8), img_saturation[300:, :]), axis=0)


        _, img_threshold = cv2.threshold(img_close, 150, 255, cv2.THRESH_BINARY)
        _, img_threshold_otsu = cv2.threshold(np.array(img_close, dtype=np.uint8), 200, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imb_blur = cv2.GaussianBlur(img_close, (19, 19), 0)
        _, img_threshold_otsu_blur = cv2.threshold(np.array(imb_blur, dtype=np.uint8), 200, 255,
                                                   cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU) #

        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_threshold')
            cv2.imshow('img_threshold',
                       np.concatenate((img_threshold, img_threshold_otsu, img_threshold_otsu_blur), axis=1))
            # cv2.imshow('cv2.THRESH_BINARY', img_threshold)
            # cv2.imshow('cv2.THRESH_BINARY + cv2.THRESH_OTSU', img_threshold_otsu)
            # cv2.namedWindow('cv2.THRESH_BINARY + cv2.THRESH_OTSU + blur')
            # cv2.imshow('cv2.THRESH_BINARY + cv2.THRESH_OTSU + blur', img_threshold_otsu_blur)
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

    def redLineSegmentation(self, frame):

        if self.flag_imshow_on == 1:
            cv2.namedWindow('origin figure')
            cv2.imshow('origin figure', frame)
            cv2.waitKey(0)

        # convert rgb to hsv
        img_saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]

        if self.flag_imshow_on == 1:
            cv2.namedWindow('saturation_figure')
            cv2.imshow('saturation_figure', img_saturation)
            cv2.waitKey(0)

        # # start to opening the image(morpoholgy)
        # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        # img_open = cv2.morphologyEx(img_saturation, cv2.MORPH_OPEN, kernel_open)
        # if self.flag_imshow_on == 1:
        #     cv2.namedWindow('img_open')
        #     cv2.imshow('img_open', img_open)
        #     cv2.waitKey(0)

        # start to closing the image(morpoholgy)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        img_close = cv2.morphologyEx(img_saturation, cv2.MORPH_CLOSE, kernel_close)
        if self.flag_imshow_on == 1:
            cv2.namedWindow('img_close')
            cv2.imshow('img_close', img_close)
            cv2.waitKey(0)

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

class dataLoadType:

    singleImage_inst = []
    calibrate_inst = []
    image_top_view = []
    JS_lineSegmentation = []
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1
    flag_first_load_src_img = 1
    flag_first_set_codec = 1

    def __init__(self, singleImage_inst):
        self.singleImage_inst = singleImage_inst
        self.calibrate_inst = calibrate_inst
        self.bridge = CvBridge()

    def __init__(self, image_top_view, calibrate_inst, JS_lineSegmentation_inst):
        self.image_top_view = image_top_view
        self.calibrate_inst = calibrate_inst
        self.JS_lineSegmentation = JS_lineSegmentation_inst
        self.bridge = CvBridge()

    def subscribeImage(self):
        print('start to subscribe image')

        global flag_is_compressed_image
        if flag_is_compressed_image == 0:
            #rospy.init_node('dataLoadType', anonymous=True)
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
            #automatically go to the callback function : self.callback()

        elif flag_is_compressed_image == 1:
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, CompressedImage, self.callback, queue_size=1)

        else:
            print('flag_is_compressed_image is wrong')

    def stopSubscribing(self):
        print('stop subscribing image')
        #self.image_sub.shutdown()
        self.rospySubImg.unregister()

    def image2video(self):
        global fileList, num_of_image_in_database
        fileList = glob.glob(str(save_concat_result + '*.png'))
        print('path_image_database is ', str(save_concat_result + '*.png'))
        num_of_image_in_database = len(fileList)

        fileList = np.sort(fileList)
        for image in fileList:
            frame = cv2.imread(image)

            if self.flag_first_set_codec == 1:
                # set codec
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter((save_concat_result + 'output.avi'), fourcc, 5.0, frame.shape[:2][::-1])

                self.flag_first_set_codec = 0

            out.write(frame)

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

    def loadImageInFiles(self):
        global fileList, flag_set_white_lane_segmentation

        # flag == 0 means do the homography transform
        # flag == 1 means extract white lane segmentation in the source image.
        if flag_set_white_lane_segmentation == 0:
            fileList = glob.glob(path_image_database)
            # print('path_image_database is ', path_image_database)

            global num_of_image_in_database
            num_of_image_in_database = len(fileList)
            # print('what is fileList', fileList, '\n')

            global count, flag_hand_homography_on, where_to_start
            count = 10000
            fileList = np.sort(fileList)
            for image in fileList:
                count = count + 1
                if count >= where_to_start:
                    if flag_hand_homography_on == 1:
                        print('the image is ', image)
                        cv2.imshow(image, cv2.imread(image))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
        else:
            self.image_src = cv2.imread(path_src_image)
            self.image_src_mask = self.JS_lineSegmentation.whiteLineSegmentation(frame=self.image_src)
            # self.image_src_mask = self.JS_lineSegmentation.redLineSegmentation(frame=self.image_src)

    def loadVideoInFiles(self):
        cap =cv2.VideoCapture('video2.mp4')

        while(cap.isOpened()):
            ret, frame = cap.read()
            self.JS_lineSegmentation.redLineSegmentation(frame=cv2.flip(src=cv2.resize(frame[0:890, :, :], dsize=(0,0), fx=0.5, fy=0.5), flipCode=-1))


            cv2.imshow('video.jpg', frame)
            cv2.waitKey(1)

    def callback(self, data):
        #print('come to callback function')
        global count
        try:
            count = count + 1

            global flag_is_compressed_image
            if flag_is_compressed_image == 0:
                # parse message into image
                # bgr8: CV_8UC3, color image with blue-green-red color order and 8bit
                self.singleImage_inst.saveImage(self.bridge.imgmsg_to_cv2(data, "bgr8"))

            elif flag_is_compressed_image == 1:
                np_arr = np.fromstring(data.data, np.uint8)
                self.singleImage_inst.saveImage(cv2.imdecode(np_arr, cv2.IMREAD_COLOR)) # cv2.CV_LOAD_IMAGE_COLOR is out of version

            else:
                print('wrong flag_is_compressed_image')

            # if you want to work asynchronously, edit the lines below
            self.wrapper()

        except CvBridgeError as e:
            print(e)

    def publishImage(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish JS image')
        self.rospyPubImg = rospy.Publisher('calibration_Francois_fisheye/image_calibrated', Image, queue_size=10)
        # rospy.init_node('calibrated_JS_lens_fisheye', anonymous=True)
        rate = rospy.Rate(10)  # 10Hz

    def wrapper(self, frame):
        global count

        if self.flag_first_load_src_img == 1:
            self.image_src_resize = cv2.resize(cv2.imread('src.jpg'), dsize=(0, 0), fx=0.2, fy=0.2)
            self.image_src_resize_origin = cv2.resize(cv2.imread('src.jpg'), dsize=(0, 0), fx=0.2, fy=0.2)

            # binary mask of white segmentation in the source image.
            if flag_1_warp_entire_src_2_mask_src == 2:
                self.image_src_mask = self.JS_lineSegmentation.whiteLineSegmentation(frame=self.image_src_resize)
                #self.image_src_mask_homography = self.calibrate_inst.homography(image_src=self.image_src_mask, image_dst=frame)

                # using mask, source image(image_src_resize) only have its white lane segmentations.
                # for row in range(self.image_src_resize.shape[0]):
                #     for col in range(self.image_src_resize.shape[1]):
                #         if self.image_src_mask[row, col] == 0:
                #             self.image_src_resize[row, col] = np.zeros((1, 3), dtype=np.uint8)
                #         else:
                #             self.image_src_resize[row, col] = np.array([0, 255, 255], dtype=np.uint8) #bgr8 type

            self.flag_first_load_src_img = 0

        if count <= 10064 or count >= 10205:
            print('fuck 01')
            # #homography test
            image_src_homography = self.calibrate_inst.homography(image_src=self.image_src_mask, image_dst=frame)

            # if flag_1_warp_entire_src_2_mask_src == 2:
            #     for row in range(frame.shape[0]):
            #         for col in range(frame.shape[1]):
            #             if image_src_mask_homography[row, col] == 0:
            #                 frame[row, col] = np.zeros((1,3), dtype=np.uint8)

            # self.JS_lineSegmentation.cornerDetect(frame, image_src_mask_homography)

            img_alphaBlended = self.calibrate_inst.alphaBlending(image_src=image_src_homography, image_dst=frame, alpha=0.5)
            cv2.imwrite((save_concat_result + str(count) + '.png'), np.concatenate((frame, img_alphaBlended), axis=1))
        else:
            cv2.imwrite((save_video_to_the_path + str(count) + '.png'), frame)

        #parking spot detection test
        # self.JS_lineSegmentation.whiteLineSegmentation(frame=cv2.resize(cv2.imread('src.jpg'),
        #  dsize=(0,0), fx=0.2, fy=0.2))
        # self.JS_lineSegmentation.whiteLineSegmentation(frame=frame)
        cv2.destroyAllWindows()

        # frame = self.JS_lineSegmentation.redLineSegmentation(
        #     frame=cv2.resize(cv2.imread('src.jpg'), dsize=(0, 0), fx=0.2, fy=0.2))
        # image_homography = self.calibrate_inst.homography(image_src=frame)
        # self.calibrate_inst.alphaBlending(image_src=image_homography,
        #                                   image_dst=cv2.resize(cv2.imread('dst.jpg'), dsize=(0, 0), fx=0.2,
        #                                                        fy=0.2), alpha=0.4)


class singleImageData:
    height = 0
    width = 0
    image = None

    def saveImage(self, img):
        self.image = img
        self.width, self.height = self.image.shape[:2][::-1]

    def resize(self, ratio):
        #cv2.resize(self.imgData, )
        print('resize of the image completed')

if __name__ == "__main__":

    print("check the version opencv.")
    print(cv2.__version__)

    #global count
    count = 0

    calibrate_inst = calibration()
    image_top_view = singleImageData()
    global flag_imshow_on, flag_print_on
    JS_lineSegmentation_inst = JS_lineSegmentation(flag_imshow_on=flag_set_imshow_on, flag_print_on=flag_set_print_on)
    dataLoadType_inst = dataLoadType(image_top_view, calibrate_inst, JS_lineSegmentation_inst)


    #global flag_subscribe_new_image_not_load_old_image
    try:
        # dataLoadType_inst.image2video()
        dataLoadType_inst.loadImageInFiles()
        # dataLoadType_inst.loadVideoInFiles()

    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()






