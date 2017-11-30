#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import cv2

# from std_msgs.msg import String
from future_car_capstone.srv import *
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import rospy
import roslib


from tempfile import TemporaryFile

import glob
import os

from cctvView import KalmanFilterClass, Import_cctvView  # , LineSegClass

from ImageClass import *
from Calibration import *
from KalmanFilter import *
from ViewTransform import *
from LineSegmentation import *

# flag
flag_fisheye_to_undistort = 1
flag_undistort_to_homography = 1
flag_homography_to_warpAffine = 1

flag_1_fisheye_2_non_fisheye = 1

flag_1_subscribeImg_2_loadImgFile = 2
flag_is_compressed_image = 1

flag_saveImg = 1
flag_publishImg = 0

flag_1_go_wrapper_2_save_subImage = 1  ####################################################
flag_saveImg_which_detect_checkerboard = 0

flag_load_detected_result = 0
flag_print = 1

# parameter
ROS_TOPIC = 'remote/image_color/compressed'
topic_to_service_in_carInfo = 'desktop/carInfo'
topic_to_service_out_carInfo = 'remote/carInfo'
path_load_image = []


def setParam():
    global path_load_image, path_save_image, path_cctv_homograph_image, cameraLocation_in_carView

    # non_fisheye
    if flag_1_fisheye_2_non_fisheye == 2:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/'
        path_cctv_homograph_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/cctvView_homography/*.png'
        cameraLocation_in_carView = (200, 250)

        if flag_undistort_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/carView/*.png'
        elif flag_homography_to_warpAffine == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/carView_homography/*.png'

    # fisheye
    elif flag_1_fisheye_2_non_fisheye == 1:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/'
        path_cctv_homograph_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView_homography/*.png'
        cameraLocation_in_carView = (253, 463)  # (195, 271)

        if flag_fisheye_to_undistort == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView/*.png'
        elif flag_undistort_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView/*.png'
        elif flag_homography_to_warpAffine == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView_homography/*.png'


nameOf_pickle_Checkerboard_Detection = 'detect_result_jaesung_171021_1600_delete_files.pickle'

fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0

# parameter from cctvView.py
shape_imgHomography = (616, 565)  # (580, 481) #

srcPixel_image = np.array([
    # [205, 946], [656 ,918], [1116, 936],
    [207, 748], [435, 731], [653, 723], [875, 724], [1114, 732],
    [202, 684], [358, 672], [505, 663], [652, 659], [800, 658], [952, 600],
    [318, 643], [433, 636], [542, 631], [652, 628], [760, 625], [876, 626],
    [475, 615], [564, 612], [652, 609], [740, 607],
    [504, 600], [579, 599], [652, 595], [724, 594]

])

dstPixel_ground = np.array([
    # [90, 225], [135, 225], [180, 225],
    [45, 180], [90, 180], [135, 180], [180, 180], [225, 180],
    [0, 135], [45, 135], [90, 135], [135, 135], [180, 135], [225, 135],
    [0, 90], [45, 90], [90, 90], [135, 90], [180, 90], [225, 90],
    [45, 45], [90, 45], [135, 45], [180, 45],
    [45, 0], [90, 0], [135, 0], [180, 0]
])









class dataLoadClass:
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1

    def __init__(self, imgInst, calibInst=None):
        self.imgInst = imgInst
        self.calibInst = calibInst
        self.bridge = CvBridge()
        self.viewTransformInst = ViewTransform(flag=flag_1_subscribeImg_2_loadImgFile, cctvViewPath=path_cctv_homograph_image)
        self.homographyInst = Homography()
        if flag_undistort_to_homography == 1:
            global srcPixel_image, dstPixel_ground, shape_imgHomography
            self.homographyInst.setHomogrphay(srcPixel_image, dstPixel_ground, shape_imgHomography, shift=(170, 180))

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

    def loadImgInFolder(self):
        global fileList
        fileList = glob.glob(path_load_image)
        # print('>>>>> path_load_image is ', path_load_image)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        # print('what is fileList', fileList, '\n')

        global count
        fileList = np.sort(fileList)
        for i in fileList:
            count = count + 1
            # print('The ', count, '-th image is ', i)
            self.imgInst.saveImage(cv2.imread(i))

            self.wrapper(nameOfFile=i)
            # tmp = cv2.imread(i)
            # print('save the roi : _' + path_save_image + 'roi/' + i[-10:])
            # cv2.imwrite(str(path_save_image + 'roi/' + i[-10:]), cv2.resize(tmp[190:440, 260:510], (0,0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))

    def publishImg(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        self.rospyPubImg = rospy.Publisher('remote/image_annotated', Image, queue_size=10)
        rospy.Rate(10)  # 10Hz

    def serviceInCarInfo(self):
        rospy.wait_for_service('desktop/carInfo')
        try:
            carInfo_Request = rospy.ServiceProxy(topic_to_service_out_carInfo, carInfo)
            carInfoResult = carInfo_Request(0)
            # print('result of the service is', (carInfoResult.carLocation_width, carInfoResult.carLocation_height), carInfoResult.carAngle)
            return (carInfoResult.carLocation_width, carInfoResult.carLocation_height), carInfoResult.carAngle
        except rospy.ServiceException as e:
            print('receiving service got error', e)
            
    def serviceOutCarInfo(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish CarLocation, CarAngle')
        self.rospyServiceCarInfo = rospy.Service(topic_to_service_out_carInfo, carInfo, self.carInfo_Request)

    def carInfo_Request(self, req):
        print('carInfo_Request : ', (self.carLocation[0], self.carLocation[1]), self.carAngle)
        return carInfoResponse(0, 0, 0, self.calibInst.homography, self.imgInst.width, self.imgInst.height)

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

            global flag_1_go_wrapper_2_save_subImage
            if flag_1_go_wrapper_2_save_subImage == 0:
                tmp = self.imgInst.imgData
                cv2.imwrite((path_save_image + 'carView/' + str((count + 10000)) + '.png'), tmp)
                cv2.imshow('hello', tmp)
                cv2.waitKey(0)

            elif flag_1_go_wrapper_2_save_subImage == 1:
                # if you want to work asynchronously, edit the lines below
                self.wrapper()

            else:
                print('wrong flag_go_to_wrapper_or_save_sub_Images')

        except CvBridgeError as e:
            print(e)

    def wrapper(self, nameOfFile=None):
        global flag_fisheye_to_undistort, flag_undistort_to_homography, flag_publishImg, flag_saveImg

        self.calibInst.height = self.imgInst.height
        self.calibInst.width = self.imgInst.width

        # undistort
        if flag_fisheye_to_undistort == 1:
            if self.flag_first_didLoadVarCalibration == 1:
                self.calibInst.loadVarAfterCalibration()

                # do not come here again
                self.flag_first_didLoadVarCalibration = 0

            self.imgInst.imgUndistort = self.calibInst.startUndistort(self.imgInst.imgData)

        elif flag_fisheye_to_undistort == 0:
            self.imgInst.imgUndistort = self.imgInst.imgData

        else:
            print('wrong flag_fisheye_to_undistort')

        # homography
        if flag_undistort_to_homography == 1:
            self.imgInst.imgHomography = self.homographyInst.startHomography(self.imgInst.imgUndistort)
        else:
            self.imgInst.imgHomography = self.imgInst.imgUndistort

        # warp
        if flag_homography_to_warpAffine == 1:
            self.viewTransformInst.getCarLocationAngle()
            # self.imgInst.imgWarp = np.zeros(shape_imgHomography, dtype=np.uint32)
            self.imgInst.imgWarp = self.viewTransformInst.generate_car_roi_in_cctvView(self.imgInst.imgHomography,
                                                                                       emptyFrame=self.imgInst.imgWarp,
                                                                                       shape_imgHomography=shape_imgHomography)
        else:
            self.imgInst.imgWarp = self.imgInst.imgHomography

        if flag_publishImg == 1:
            try:
                self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(self.imgInst.imgWarp, "bgr8"))
            except CvBridgeError as e:
                print(e)
        else:
            cv2.imshow('self.imgInst.imgWarp', self.imgInst.imgWarp)
            cv2.waitKey(1)

        if flag_saveImg == 1 and flag_1_subscribeImg_2_loadImgFile == 2:
            # cv2.imwrite(path_save_image + 'carView/' + nameOfFile[-10:], self.imgInst.imgData)

            if flag_fisheye_to_undistort == 1:
                cv2.imwrite(path_save_image + 'carView_undistort/' + nameOfFile[-10:], self.imgInst.imgUndistort)

            if flag_undistort_to_homography == 1:
                cv2.imwrite(path_save_image + 'carView_homography/' + nameOfFile[-10:], self.imgInst.imgHomography)

            if flag_homography_to_warpAffine == 1:
                # print('save iamge with name : ', path_save_image + 'carView_2_cctvView/' + nameOfFile[-10:])
                cv2.imwrite(path_save_image + 'carView_2_cctvView/' + nameOfFile[-10:], self.imgInst.imgWarp)

        elif flag_saveImg == 1 and flag_1_subscribeImg_2_loadImgFile == 1:
            cv2.imwrite((path_save_image + 'carView/' + str(100000 + count) + '.png'), self.imgInst.imgData)

            if flag_fisheye_to_undistort == 1:
                cv2.imwrite((path_save_image + 'carView_undistort/' + str(100000 + count) + '.png'), self.imgInst.imgUndistort)

            if flag_undistort_to_homography == 1:
                cv2.imwrite((path_save_image + 'carView_homography/' + str(100000 + count) + '.png'), self.imgInst.imgHomography)

            if flag_homography_to_warpAffine == 1:
                # print('save iamge with name : ', path_save_image + 'carView_2_cctvView/' + nameOfFile[-10:])
                cv2.imwrite((path_save_image + 'carView_2_cctvView/' + str(100000 + count) + '.png'), self.imgInst.imgWarp)



class carView_import:
    def __init__(self):
        print('init carView_import')

    def setCarViewHomography(self):
        global srcPixel_image, dstPixel_ground
        shift = (170, 180)
        for i in range(dstPixel_ground.shape[0]):
            dstPixel_ground[i][0] = dstPixel_ground[i][0] + shift[0]
            dstPixel_ground[i][1] = dstPixel_ground[i][1] + shift[1]
        return cv2.findHomography(srcPoints=dstPixel_ground, dstPoints=srcPixel_image, method=cv2.RANSAC)

    def setCarViewShape(self):
        return (1288, 964)

if __name__ == "__main__":

    print("check the version opencv.")
    print(cv2.__version__)

    # global count
    # global -> error here
    # count = 0
    setParam()
    imgInst = ImageClass()
    calibInst = Calibration()
    dataLoadInst = dataLoadClass(imgInst, calibInst)

    # global flag_1_subscribeImg_2_loadImgFile
    try:
        if flag_1_subscribeImg_2_loadImgFile == 1:
            # One python file for one init_node
            rospy.init_node('carView', anonymous=True)
            dataLoadInst.subscribeImg()

            if flag_publishImg == 1:
                dataLoadInst.publishImg()
                # while(True):
                #     dataLoadInst.serviceInCarInfo()

            rospy.spin()

        elif flag_1_subscribeImg_2_loadImgFile == 2:
            if flag_publishImg == 1:
                rospy.init_node('carView', anonymous=True)
                dataLoadInst.publishImg()

            dataLoadInst.loadImgInFolder()

    except KeyboardInterrupt:
        print("Shutting down")
