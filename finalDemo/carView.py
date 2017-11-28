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

import pickle
from tempfile import TemporaryFile

import glob
import os

from cctvView import KalmanFilterClass, Import_cctvView #, LineSegClass

#flag
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



#parameter
ROS_TOPIC = 'remote/image_color/compressed'
path_load_image = []
def setParam():
    global path_load_image, path_save_image, path_cctv_homograph_image, cameraLocation_in_carView

    #non_fisheye
    if flag_1_fisheye_2_non_fisheye == 2:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/'
        path_cctv_homograph_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/cctvView_homography/*.png'
        cameraLocation_in_carView = (200, 250)

        if flag_undistort_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/carView/*.png'
        elif flag_homography_to_warpAffine == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/carView_homography/*.png'

    #fisheye
    elif flag_1_fisheye_2_non_fisheye == 1:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/'
        path_cctv_homograph_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView_homography/*.png'
        cameraLocation_in_carView = (199, 461) #(195, 271)

        if flag_fisheye_to_undistort == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView/*.png'
        elif flag_undistort_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView/*.png'
        elif flag_homography_to_warpAffine == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/carView_homography/*.png'


nameOf_pickle_Checkerboard_Detection = 'detect_result_jaesung_171021_1600_delete_files.pickle'
path_pickle_calibration_variable = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/calib_result_JS_fisheye.pickle'
fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0


#parameter from cctvView.py
shape_imgHomography=(616, 565) #(580, 481) #

class calibClassCar:
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
        # frame=cv2.resize(src=frame, dsize=(1288, 964)) ########################################################################################################

        return cv2.fisheye.undistortImage(frame, self.camera_matrix, self.distCoeffs,
                                                       Knew=self.camera_matrix, new_size=frame.shape[:2][::-1])#(1288, 964)

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

        frame_homography_RANSAC = cv2.warpPerspective(frame, self.homography_RANSAC, (360, 300 + 180))#frame.shape[:2][::-1]

        return frame_homography_RANSAC

#CarViewHomography_2_CctvViewHomography:
class ViewTransform:
    # carLocation = np.zeros(2, dtype=np.float32)
    def __init__(self, cctvViewTopic=None, cctvViewPath=None):
        print('in the __init__ of ViewTransforms')

        #subscribe cctv frames
        if flag_1_subscribeImg_2_loadImgFile == 1:
            self.import_cctvView_inst = Import_cctvView()
            self.import_cctvView_inst.__int__()

        #load cctv frames
        elif flag_1_subscribeImg_2_loadImgFile == 2:
            self.fileList_cctv = glob.glob(cctvViewPath)
            self.fileList_cctv = np.sort(self.fileList_cctv)
            print('>>>>> self.fileList_cctv', self.fileList_cctv)
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
            # print('dataLoadInst.serviceCarInfo()', dataLoadInst.serviceCarInfo(), 'count is ', count)
            return dataLoadInst.serviceCarInfo()

    def generate_car_roi_in_cctvView(self, frameCarView, emptyFrame=None):
        #why it is error???
        # carLocation = np.zeros(2, dtype=np.int64)
        # frameCCTVView, (carLocation[0], carLocation[1]), carAngle = self.getCarLocationAngle() #carLocation = (width, height), carAngle=(theta) refer to cctvView.py
        if flag_1_subscribeImg_2_loadImgFile == 1:
            #if None .........................it encounters errors 'NoneType' object is not iterable #############################################################################################
            carLocation, carAngle = self.getCarLocationAngle() #carLocation = (width, height), carAngle=(theta) refer to cctvView.py
            frameCCTVView = None
        elif flag_1_subscribeImg_2_loadImgFile == 2:
            frameCCTVView, carLocation, carAngle = self.getCarLocationAngle()  # carLocation = (width, height), carAngle=(theta) refer to cctvView.py

        global cameraLocation_in_carView

        tmp = (carLocation[0] - cameraLocation_in_carView[0], carLocation[1] - cameraLocation_in_carView[1])
        print('tep = ', tmp, 'carLocation = ', carLocation, 'cameraLocation_incarView = ', cameraLocation_in_carView)
        matrixTranslate = np.float32([[1, 0, tmp[0]], [0, 1, tmp[1]]])
        matrixRotation = cv2.getRotationMatrix2D(center=carLocation, angle=carAngle, scale=1)#carLocation
        # print('matrixTranslate, matrixRotation, carLocation, carAngle is ')
        # print(matrixTranslate)
        # print(matrixRotation)
        # print(carLocation, carAngle)
        # matrixRotation[:][2] = matrixTranslate ##################################################################################
        # cv2.imshow('frameCarView', frameCarView)

        emptyFrame = cv2.warpAffine(src=frameCarView, M=matrixTranslate, dsize=shape_imgHomography)
        emptyFrame = cv2.warpAffine(src=emptyFrame, M=matrixRotation, dsize=shape_imgHomography)

        #draw cameraLocation_in_cctvView
        cv2.circle(emptyFrame, carLocation, radius=5, color=(0, 180, 180), thickness=4)
        if frameCCTVView is None:
            return emptyFrame.copy()

        else:
            print('shape of the frameCCTVView is ', frameCCTVView.shape[:2][::-1], emptyFrame.shape[:2][::-1])
            return self.alphaBlending(img1=frameCCTVView, img2=emptyFrame)


    def alphaBlending(self, img1, img2):
        alpha = 0.3
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
        print('>>>>> path_load_image is ', path_load_image)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        print('what is fileList', fileList, '\n')

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

    def serviceCarInfo(self):
        rospy.wait_for_service('desktop/carInfo')
        try:
            carInfo_Request = rospy.ServiceProxy('desktop/carInfo', carInfo)
            carInfoResult = carInfo_Request(0)
            # print('result of the service is', (carInfoResult.carLocation_width, carInfoResult.carLocation_height), carInfoResult.carAngle)
            return (carInfoResult.carLocation_width, carInfoResult.carLocation_height), carInfoResult.carAngle
        except rospy.ServiceException as e:
            print('receiving service got error')

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

        #undistort
        if flag_fisheye_to_undistort == 1:
            if self.flag_first_didLoadVarCalibration == 1:
                self.calibInst.loadVarAfterCalibration()

                #do not come here again
                self.flag_first_didLoadVarCalibration = 0

            self.imgInst.imgUndistort = self.calibInst.startUndistort(self.imgInst.imgData)

        elif flag_fisheye_to_undistort == 0:
            self.imgInst.imgUndistort = self.imgInst.imgData.copy()

        else:
            print('wrong flag_fisheye_to_undistort')

        #homography
        if flag_undistort_to_homography == 1:
            self.imgInst.imgHomography = self.calibInst.startHomography(self.imgInst.imgUndistort)
        else:
            self.imgInst.imgHomography = self.imgInst.imgUndistort.copy()

        #warp
        if flag_homography_to_warpAffine == 1:
            self.viewTransformInst.getCarLocationAngle()
            # self.imgInst.imgWarp = np.zeros(shape_imgHomography, dtype=np.uint32)
            self.imgInst.imgWarp = self.viewTransformInst.generate_car_roi_in_cctvView(self.imgInst.imgHomography, self.imgInst.imgWarp)
        else:
            self.imgInst.imgWarp = self.imgInst.imgHomography.copy()

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


class imgClass:
    height = 0
    width = 0
    imgData = None
    imgUndistort = None
    imgHomography = None
    imgWarp = None

    def saveImage(self, img):
        if (964 != img.shape[0]) or (1288 != img.shape[1]): #(1288, 964) != img.shape[:2][::-1]
            print('begin reshape the self.imgInst.imgData')
            self.imgData = cv2.resize(src=img, dsize=(1288,964))
        else:
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
    calibInst = calibClassCar()
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
                # while(True):
                #     dataLoadInst.serviceCarInfo()

            rospy.spin()

       elif flag_1_subscribeImg_2_loadImgFile == 2:
           if flag_publishImg == 1:
               rospy.init_node('carView', anonymous=True)
               dataLoadInst.publishImg()

           dataLoadInst.loadImgInFolder()



    except KeyboardInterrupt:
        print("Shutting down")
