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

from cctvView import KalmanFilterClass, Import_cctvView #, LineSegClass

#flag
flag_fisheye_to_undistort = 0
flag_undistort_to_homography = 0
flag_homography_to_warpAffine = 1

flag_1_fisheye_2_non_fisheye = 2

flag_1_subscribeImg_2_loadImgFile = 2
flag_is_compressed_image = 1

flag_saveImg = 1
flag_publishImg = 0




flag_go_to_wrapper_or_save_subImage = 0
flag_saveImg_which_detect_checkerboard = 0

flag_load_detected_result = 0
flag_print = 1



#parameter
ROS_TOPIC = 'remote/image_color/compressed'#'jaesung_lens_camera/image_color'
path_load_image = []
def setParam():
    global path_load_image, path_save_image, path_cctv_homograph_image
    if flag_1_fisheye_2_non_fisheye == 2:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/'
        path_cctv_homograph_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/cctvView_homography/*.png'

        if flag_undistort_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/carView/*.png'
        elif flag_homography_to_warpAffine == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/carView_homography/*.png'

    elif flag_1_fisheye_2_non_fisheye == 1:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/'
        path_cctv_homograph_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView_homography/*.png'

        if flag_undistort_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView/*.png'
        elif flag_homography_to_warpAffine == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView_homography/*.png'
        elif flag_fisheye_to_undistort == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView/*.png'


nameOf_pickle_Checkerboard_Detection = 'detect_result_jaesung_171021_1600_delete_files.pickle'
path_pickle_calibration_variable = '/home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/calib_result_JS_fisheye.pickle'
fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0

#parameter from cctvView.py
shape_imgHomography=(616, 565)
cameraLocation_in_carView = (200, 250)

class calibClass:
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
    height = 0
    width = 0

    def loadVarAfterCalibration(self):
        with open(path_pickle_calibration_variable) as f:
            self.camera_matrix, self.distCoeffs, self.new_camera_matrix, self.width_trainImage, self.height_trainImage, self.roi, self.map1, self.map2 = pickle.load(f)

        global flag_print
        if flag_print == 1:
            print('camera matrix is ')
            print(self.camera_matrix)
            print('new camera Matrix is ')
            print(self.new_camera_matrix)
            print('distort Coeffs is ')
            print(self.distCoeffs)
            print('width, height is ', self.width_trainImage, self.height_trainImage)

    def startUndistort(self, frame):
        # dim0 = frame.shape[:2][::-1] #(width, height)
        # dim0 = (self.width_trainImage, self.height_trainImage)

        return cv2.fisheye.undistortImage(frame, self.camera_matrix, self.distCoeffs,
                                                       Knew=self.camera_matrix, new_size=frame.shape[:2][::-1])#dim0

    def startHomography(self, frame):

        if self.flag_first_didHomography == 1:

            srcPixel_image = np.array([
                [288, 682], [400, 682], [525, 681], [663, 675], [802, 667], [ 932, 658],
                [197, 713], [327, 715], [476, 716], [655, 711], [817, 700], [ 980, 681],
                [117, 742], [255, 750], [435, 754], [646, 752], [850, 737], [1053, 715],
                [ 12, 776], [157, 800], [367, 813], [637, 815], [906, 798], [1146, 761],
                            [ 10, 868], [251, 917], [615, 934], [990, 905], [1279, 840]

            ])

            dstPixel_ground = np.array([
                            [80,   0], [120,   0], [160,   0], [200,   0], [240,   0], [280,   0]
                          , [80,  40], [120,  40], [160,  40], [200,  40], [240,  40], [280,  40]
                          , [80,  80], [120,  80], [160,  80], [200,  80], [240,  80], [280,  80]
                          , [80, 120], [120, 120], [160, 120], [200, 120], [240, 120], [280, 120]
                                      ,[120, 160], [160, 160], [200, 160], [240, 160], [280, 160]
            ])

            # srcPixel_image, dstPixel_ground = np.array(srcPixel_image), np.array(dstPixel_ground)

            self.homography_RANSAC, mask = cv2.findHomography(srcPoints=srcPixel_image,
                                                              dstPoints=dstPixel_ground,
                                                              method=0)#cv2.RANSAC
            print('homography is ', self.homography_RANSAC)

            #do not calculate homography again
            self.flag_first_didHomography = 0

        frame_homography_RANSAC = cv2.warpPerspective(frame, self.homography_RANSAC, (360, 300))#frame.shape[:2][::-1]

        return frame_homography_RANSAC

#CarViewHomography_2_CctvViewHomography:
class ViewTransform:
    # carLocation = np.zeros(2, dtype=np.float32)
    def __init__(self, cctvViewTopic=None, cctvViewPath=None):
        print('in the __init__ of ViewTransforms')
        if flag_1_subscribeImg_2_loadImgFile == 1:
            print('later')

        elif flag_1_subscribeImg_2_loadImgFile == 2:
            self.fileList_cctv = glob.glob(cctvViewPath)
            self.fileList_cctv = np.sort(self.fileList_cctv)
            print('self.fileList_cctv', self.fileList_cctv)
            # print('>> is the class Import_cctvView initialized?')
            self.import_cctvView_inst = Import_cctvView()
            # print('>> is the class Import_cctvView initialized? __init__ is needed ?') #yes
            self.import_cctvView_inst.__int__()

    def getCarLocationAngle(self):
        # print('find location and angle of the car in cctvView')
        if flag_1_subscribeImg_2_loadImgFile == 2:
            # location = np.zeros(2, dtype=np.float32)
            global count
            return self.import_cctvView_inst.getCarLocationAngle(frame_cctv=cv2.imread(self.fileList_cctv[count]))#[]##########################################################

        elif flag_1_subscribeImg_2_loadImgFile == 1:
            print('not finalize coding yet')

    def generate_car_roi_in_cctvView(self, frameCarView, emptyFrame=None):
        frameCCTVView, carLocation, carAngle = self.getCarLocationAngle() #carLocation = (width, height), carAngle=(theta) refer to cctvView.py
        global cameraLocation_in_carView

        matrixTranslate = np.float32([[1, 0, carLocation[0] - cameraLocation_in_carView[0]], [0, 1, carLocation[1] - cameraLocation_in_carView[1]]])
        matrixRotation = cv2.getRotationMatrix2D(center=carLocation, angle=carAngle, scale=1)
        # print('matrixTranslate, matrixRotation, carLocation, carAngle is ')
        # print(matrixTranslate)
        # print(matrixRotation)
        # print(carLocation, carAngle)
        # matrixRotation[:][2] = matrixTranslate ##################################################################################
        # cv2.imshow('frameCarView', frameCarView)

        emptyFrame = cv2.warpAffine(src=frameCarView, M=matrixTranslate, dsize=shape_imgHomography)
        emptyFrame = cv2.warpAffine(src=emptyFrame, M=matrixRotation, dsize=shape_imgHomography)

        #draw cameraLocation_in_cctvView
        cv2.circle(emptyFrame, carLocation, radius=5, color=(0, 0, 255), thickness=4)
        return self.alphaBlending(img1=frameCCTVView, img2=emptyFrame)

    def alphaBlending(self, img1, img2):
        alpha = 0.7
        beta = (1 - alpha)
        return cv2.addWeighted(src1=img1, alpha=alpha, src2=img2, beta=beta, gamma=0.0, dst=None, dtype=-1)

class dataLoadClass:

    imgInst = []
    calibInst = []
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1

    def __init__(self, imgInst, calibInst=None, viewTransformInst=None):
        self.imgInst = imgInst
        self.calibInst = calibInst
        self.bridge = CvBridge()
        self.viewTransformInst = viewTransformInst

    def subscribeImg(self):
        print('start to subscribe image')

        if flag_is_compressed_image == 0:
            #rospy.init_node('dataLoadClass', anonymous=True)
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
            #automatically go to the callback function : self.callback()

        elif flag_is_compressed_image == 1:
            self.rospySubImg = rospy.Subscriber(ROS_TOPIC, CompressedImage, self.callback, queue_size=1)

        else:
            print('flag_is_compressed_image is wrong')

    def loadImgInFolder(self):
        global fileList
        fileList = glob.glob(path_load_image)
        # print('path_load_image is ', path_load_image)

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
                self.imgInst.saveImage(cv2.imdecode(np_arr, cv2.IMREAD_COLOR)) # cv2.CV_LOAD_IMAGE_COLOR is out of version

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
        global flag_fisheye_to_undistort, flag_undistort_to_homography, flag_publishImg, flag_saveImg

        self.calibInst.height = self.imgInst.height
        self.calibInst.width = self.imgInst.width

        if flag_fisheye_to_undistort == 1 and flag_undistort_to_homography == 1:
            # load Calibration results and show undistorted image
            if self.flag_first_didLoadVarCalibration == 1:
                self.calibInst.loadVarAfterCalibration()

                # do not come here again
                self.flag_first_didLoadVarCalibration = 0
            
            self.imgInst.imgUndistort = self.calibInst.startUndistort(self.imgInst.imgData)
            self.imgInst.imgHomography = self.calibInst.startHomography(self.imgInst.imgUndistort)

        elif flag_fisheye_to_undistort == 0 and flag_undistort_to_homography == 1:
            self.imgInst.imgHomography = self.calibInst.startHomography(self.imgInst.imgData)
            cv2.imshow('self.imgInst.imgHomography', np.concatenate((self.imgInst.imgHomography, cv2.resize(self.imgInst.imgData, (300, 300))), axis=1))
            cv2.waitKey(1)

        elif flag_fisheye_to_undistort == 0 and flag_undistort_to_homography == 0 and flag_homography_to_warpAffine == 1:
            self.viewTransformInst.getCarLocationAngle()
            self.imgInst.imgWarp = np.zeros(shape_imgHomography, dtype=np.uint32)
            self.imgInst.imgWarp = self.viewTransformInst.generate_car_roi_in_cctvView(self.imgInst.imgData, self.imgInst.imgWarp)
            cv2.imshow('viewTransform display', self.imgInst.imgWarp)
            cv2.waitKey(1)



        if flag_publishImg == 1:
            try:
                self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(self.imgInst.imgUndistort, "bgr8"))
            except CvBridgeError as e:
                print(e)
                
        if flag_saveImg == 1:
            if flag_fisheye_to_undistort == 0 and flag_undistort_to_homography == 1:
                # print('save iamge with name : ', path_save_image + 'carView_homography/' + nameOfFile[-9:])
                cv2.imwrite(path_save_image + 'carView_homography/' + nameOfFile[-9:], self.imgInst.imgHomography)

            elif flag_fisheye_to_undistort == 0 and flag_undistort_to_homography == 0 and flag_homography_to_warpAffine == 1:
                # print('save iamge with name : ', path_save_image + 'carView_2_cctvView/' + nameOfFile[-9:])
                cv2.imwrite(path_save_image + 'carView_2_cctvView/' + nameOfFile[-9:], self.imgInst.imgWarp)

class imgClass:
    height = 0
    width = 0
    imgData = None
    imgUndistort = None
    imgHomography = None
    imgWarp = None

    def saveImage(self, img):
        self.imgData = img
        self.height, self.width = self.imgData.shape[:2]

if __name__ == "__main__":

    print("check the version opencv.")
    print(cv2.__version__)

    # global count
    # global -> error here
    # count = 0
    setParam()
    imgInst = imgClass()
    calibInst = calibClass()
    if flag_homography_to_warpAffine == 1:
        # global path_cctv_homograph_image # no global in here
        viewTransformInst = ViewTransform(cctvViewPath=path_cctv_homograph_image)
    else:
        viewTransformInst = None

    dataLoadInst = dataLoadClass(imgInst, calibInst, viewTransformInst)

    #global flag_1_subscribeImg_2_loadImgFile
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
