#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import rospy
import roslib

import pickle
from tempfile import TemporaryFile

import glob
import os

# flag
flag_1_subscribeImg_2_loadImgFile = 2
flag_publishImg = 0

flag_saveImg = 1
flag_1_undistort_2_homography_3_original = 1

flag_is_compressed_image = 1
flag_go_to_wrapper_or_save_subImage = 0

# flag which does not need fixing anymore
flag_saveImg_which_detect_checkerboard = 0
flag_load_calibrated_result = 1
flag_load_detected_result = 0
flag_print = 1

# parameter
ROS_TOPIC = 'remote/image_color/compressed'
path_load_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView/*.png'
path_save_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView_homography/'

nameOf_pickle_Checkerboard_Detection = 'detect_result_jaesung_171021_1600_delete_files.pickle'
path_pickle_calibration_variable = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/calib_result_JS_fisheye.pickle'
fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0
shape_imgHomography=(616, 565)


class calibClass:
    flag_first_didHomography = 1

    def startHomography(self, frame_JS):

        if self.flag_first_didHomography == 1:
            # cctvView to ground
            srcPixel_Img = np.array([
                [575, 138], [653, 117], [746, 93],
                [540, 269], [623, 255], [720, 237],

                [552, 544], [650, 547]
            ])

            dstPixel_ground = np.array([
                [575, 0], [145, 0], [235, 0],
                [55, 155], [145, 155], [235, 155],

                [145, 405], [235, 405]
            ])

            self.homography_RANSAC, mask = cv2.findHomography(srcPoints=srcPixel_Img,
                                                              dstPoints=dstPixel_ground,
                                                              method=cv2.RANSAC)
            print('homography is ', self.homography_RANSAC)

            # do not calculate homography again
            self.flag_first_didHomography = 0

        global shape_imgHomography
        frame_homography_RANSAC = cv2.warpPerspective(frame_JS, self.homography_RANSAC, shape_imgHomography)

        # cv2.namedWindow('Transformation of undistorted frame of JS fisheye camera using homography')
        # cv2.imshow('Transformation of undistorted frame of JS fisheye camera using homography', frame_homography_RANSAC)
        # cv2.waitKey(1)

        return frame_homography_RANSAC  # frame_homography_JS


class dataLoadClass:
    imgInst = []
    calibInst = []
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1

    def __init__(self, imgInst, calibInst):
        self.imgInst = imgInst
        self.calibInst = calibInst
        self.bridge = CvBridge()

    def subscribeImg(self):
        print('start to subscribe image')

        if flag_is_compressed_image == 0:
            # rospy.init_node('dataLoadClass', anonymous=True)
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
            # automatically go to the callback function : self.callback()

        elif flag_is_compressed_image == 1:
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, CompressedImage, self.callback, queue_size=1)

        else:
            print('flag_is_compressed_image is wrong')

    def stopSubscribing(self):
        print('stop subscribing image')
        # self.image_sub.shutdown()
        self.rospySubImg.unregister()

    def loadImgInFolder(self):
        global fileList
        fileList = glob.glob(path_load_image)
        #print('path_load_image is ', path_load_image)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        #print('what is fileList', fileList, '\n')

        global count
        for i in fileList:
            count = count + 1
            # print('The ', count, '-th image is ', i)
            self.imgInst.saveImage(cv2.imread(i))

            self.wrapper(nameOfFile=i)
            # cv2.imshow(i, cv2.imread(i))
            # cv2.waitKey(0)
            # print('save the roi : _' + path_save_image + 'roi/' + i[-9:])
            # cv2.imwrite(str(path_save_image + 'roi/' + i[-9:]), cv2.resize(tmp[190:440, 260:510], (0,0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))

    def publishImg(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish JS image')
        self.rospyPubImg = rospy.Publisher('calibration_JS_fisheye/image_calibrated', Image, queue_size=10)
        rospy.Rate(10)  # 10Hz

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
                cv2.imwrite((path_save_image + str((count + 10000)) + '.png'), tmp)
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
        global count, num_of_image_in_database
        global flag_load_calibrated_result

        self.imgInst.imgHomography = self.calibInst.startHomography(self.imgInst.imgData)

        if flag_publishImg == 1:
            try:
                self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(self.imgInst.imgHomography, "bgr8"))
            except CvBridgeError as e:
                print(e)

        if flag_saveImg == 1:
            print('save iamge with name : ', path_save_image + nameOfFile[-9:])
            cv2.imwrite(path_save_image + nameOfFile[-9:], self.imgInst.imgHomography)
            # cv2.imwrite(path_save_image + 'homography/' + nameOfFile[:-9], self.imgInst.imgHomography)


class imgClass:
    height = 0
    width = 0
    imgData = None
    imgUndistort = None
    imgHomography = None

    def saveImage(self, img):
        self.imgData = img
        self.height, self.width = self.imgData.shape[:2]


if __name__ == "__main__":

    print("check the version opencv.")
    print(cv2.__version__)

    # global count
    # global -> error here
    # count = 0

    imgInst = imgClass()
    calibInst = calibClass()
    dataLoadInst = dataLoadClass(imgInst, calibInst)

    # global flag_1_subscribeImg_2_loadImgFile
    try:
        if flag_1_subscribeImg_2_loadImgFile == 1:
            # One python file for one init_node
            rospy.init_node('carView', anonymous=True)
            dataLoadInst.subscribeImg()

            if flag_publishImg == 1:
                dataLoadInst.publishImg()

            rospy.spin()

        elif flag_1_subscribeImg_2_loadImgFile == 2:
            dataLoadInst.loadImgInFolder()

            if flag_publishImg == 1:
                dataLoadInst.publishImg()

    except KeyboardInterrupt:
        print("Shutting down")
