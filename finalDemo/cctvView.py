#!/usr/bin/env python

# from __future__ import print_function
import numpy as np
import cv2

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from future_car_capstone.srv import *
from future_car_capstone.msg import *
# impot roslib.load_manifest future_car_capstone.
from cv_bridge import CvBridge, CvBridgeError

import rospy
import glob
import time

from yolo import MyDarknet
from ImageClass import *
from Calibration import *
from KalmanFilter import *
from ViewTransform import *
from LineSegmentation import *
from carView import carView_import

# from carView import calibClassCar,ViewTransform

# flag
flag_1_subscribeImg_2_loadImgFile_3_image2video = 2  ######################
flag_origin_to_homography = 0
flag_homography_to_drawCircle = 1
flag_find_empty_parking_lot = 0
flag_drawCircle_to_carView = 0


flag_is_YOLO = 1

flag_1_fisheye_2_non_fisheye = 1
flag_1_measureONLY_2_kalman = 2

flag_saveImg = 1
flag_publishImg = 1

flag_is_compressed_image = 1
flag_go_to_wrapper_or_save_subImage = 1  #################################

# flag which does not need fixing anymore
flag_saveImg_which_detect_checkerboard = 0
flag_load_calibrated_result = 1
flag_load_detected_result = 0
flag_print = 0

# parameter
cctvLocation3D = (235, 520, 530) ##########################################################################
carLocationDepth = 30

topic_to_subscribe_image = 'desktop/image_color/compressed'
topic_to_publish_image = 'desktop/image_annotated'
topic_to_service_out_carInfo = 'desktop/carInfo'
topic_to_service_in_carInfo = 'remote/carInfo'
Qfactor = 0.1
QfactorProcessNoise = 0.001

path_load_image = []
path_save_image = []
path_carView_undistort_image = []
kernelSize_close = (50, 50)
kernelSize_open = (30, 30)


def setParam():
    global path_load_image, path_save_image, kernelSize_close, kernelSize_open
    global flag_origin_to_homography, flag_homography_to_drawCircle, path_carView_undistort_image

    # non_fisheye
    if flag_1_fisheye_2_non_fisheye == 2:
        path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/'
        kernelSize_close = (50, 50)
        kernelSize_open = (30, 30)

        if flag_origin_to_homography == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/cctvView/*.png'
        elif flag_homography_to_drawCircle == 1:
            path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/non_fisheye_lens/cctvView_homography/*.png'

    # fisheye
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

fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0


shape_imgHomography = (620, 800)
#(550, 480) #partially erased
#(610, 565) is the all parking lot, 510 is only with visible area

srcPixel_Img = np.array([
    # i do not know
    [148,  48], [201,  45], [259,  42], [437,  48],
    [124, 123], [184, 121], [247, 120], [447, 124],

    [ 88, 295], [153, 300],
    [ 78, 369]

    # # 171202 data
    # [165, 246], [200, 242], [238, 238], [367, 223],
    # [146, 298], [183, 296], [224, 293], [370, 282],
    #
    # [109, 413], [150, 418],
    # [ 96, 463]

    # # 171128 data
    # [145, 117], [193, 113], [245, 108], [418, 104],
    # [119, 190], [169, 187], [224, 185], [418, 185],
    #
    # [74, 342], [124, 350],
    # [57, 414]

    # # 1288 * 964, 171122 data
    # [286, 70], [326, 60], [370, 47],
    # [268, 136], [312, 129], [358, 121],
    #
    # [274, 276], [323, 278]

    # ## 171125
    # [138, 70], [181, 63], [232, 56], [401, 46],
    # [112, 142], [158, 137], [211, 131], [399, 124],
    #
    # [70, 291], [118, 294],
    # [56, 358]
])

dstPixel_ground = np.array([

    # # # 171128 data
    [55,   0], [145,   0], [235,   0], [508,   0],
    [55, 155], [145, 155], [235, 155], [508, 155],

    [55, 405], [145, 405],
    [55, 510]


    # # 1288 * 964, 171122 data
    # [55, 0], [145, 0], [235, 0],
    # [55, 155], [145, 155], [235, 155],
    # [145, 405], [235, 405]


    # ## 171125
    # [55, 0], [145, 0], [235, 0], [508, 0],
    # [55, 155], [145, 155], [235, 155], [508, 155],
    #
    # [55, 405], [145, 405],
    # [55, 510]
])

parkingLot = np.array([
    [82, 59], [190, 65], [282, 51], [374, 55], [460, 62],
    [98, 426], [200, 429], [300, 424]
])

class DataLoadClass:
    imgInst = []
    calibInst = []
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1
    flag_first_set_homography_ground_to_carView = 1

    def __init__(self, imgInst, calibInst, homographyInst=None):
        self.imgInst = imgInst
        self.calibInst = calibInst
        self.bridge = CvBridge()

        global cctvLocation3D, carLocationDepth
        self.lineSegInst = LineSegClass(flag_imshow_on=1, flag_print_on=1, cctvLocation3D=cctvLocation3D, carLocationDepth=carLocationDepth)

        self.homographyInst = homographyInst

        global flag_origin_to_homography, shape_imgHomography
        if flag_origin_to_homography == 1 and self.homographyInst is not None:
            global srcPixel_Img, dstPixel_ground
            self.homographyInst.setHomogrphay(srcPixel_Img, dstPixel_ground, outputShape=shape_imgHomography, shift=None)

        global flag_is_YOLO
        if flag_is_YOLO == 1:
            self.myDarknetInst = MyDarknet()
            self.myDarknetInst.__int__()
            print('self.myDarknetInst', self.myDarknetInst)

    def subscribeImg(self):
        print('start to subscribe image')
        if flag_is_compressed_image == 0:
            # rospy.init_node('DataLoadClass', anonymous=True)
            self.rospySubImg = rospy.Subscriber(topic_to_subscribe_image, Image, self.callback)
            # automatically go to the callback function : self.callback()
            # rospy.Rate(1)  # 10Hz

        elif flag_is_compressed_image == 1:
            self.rospySubImg = rospy.Subscriber(topic_to_subscribe_image, CompressedImage, self.callback, queue_size=10)

        else:
            print('flag_is_compressed_image is wrong')

    def callback(self, data):
        global count
        count = count + 1

        try:
            start = time.time()

            if flag_is_compressed_image == 0:
                # parse message into image
                # bgr8: CV_8UC3, color image with blue-green-red color order and 8bit
                self.imgInst.saveImage(self.bridge.imgmsg_to_cv2(data, "bgr8"))

            elif flag_is_compressed_image == 1:
                np_arr = np.fromstring(data.data, np.uint8)
                self.imgInst.saveImage(cv2.imdecode(np_arr, cv2.IMREAD_COLOR))  # cv2.CV_LOAD_IMAGE_COLOR is out of version

            else:
                print('flag_is_compressed_image is wrong')

            global flag_go_to_wrapper_or_save_subImage
            if flag_go_to_wrapper_or_save_subImage == 0:
                cv2.imwrite((path_save_image + 'cctvView/' + str((count + 10000)) + '.png'), self.imgInst.imgData)
                cv2.imshow('self.imgInst.imgData', self.imgInst.imgData)
                cv2.waitKey(1)

            elif flag_go_to_wrapper_or_save_subImage == 1:
                # if you want to work asynchronously, edit the lines below
                self.wrapper()


            else:
                print('wrong flag_go_to_wrapper_or_save_sub_Images')

            end = time.time()
            seconds = end - start
            print('>>> fps = ', seconds, '(sec)')

        except CvBridgeError as e:
            print('bridge error', e)

    def loadImgInFolder(self):
        global fileList
        fileList = glob.glob(path_load_image)
        # print('path_load_image is ', path_load_image)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        fileList = np.sort(fileList)
        # print('what is fileList', fileList, '\n')

        global count
        for i in fileList:
            count = count + 1

            # print('The ', count, '-th image is ', i)
            self.imgInst.saveImage(cv2.imread(i), resize=(640, 488))
            self.wrapper(nameOfFile=i)

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

    def publishCarInfoArduino(self):
        self.rospyPubCarInfoArduino = rospy.Publisher('led_out', Float32MultiArray, queue_size=10)

    def publishImg(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish JS image')
        self.rospyPubImg = rospy.Publisher(topic_to_publish_image, Image, queue_size=10)
        # rospy.Rate(10)  # 10Hz

    def serviceCarInfo(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish CarLocation, CarAngle')
        self.rospyServiceCarInfo = rospy.Service(topic_to_service_out_carInfo, carInfo, self.carInfo_Request)

    def carInfo_Request(self, req):
        # print('carInfo_Request : ', int(self.carLocation[0]), int(self.carLocation[1]), self.carAngle, (0,0,0,0,0,0,0,0,0), self.imgInst.width, self.imgInst.height)
        return carInfoResponse(int(self.carLocation[0]), int(self.carLocation[1]), float(self.carAngle), (0,0,0,0,0,0,0,0,0), int(self.imgInst.width), int(self.imgInst.height))#tuple(np.zeros((3,3), dtype=np.float32))

    def wrapper(self, nameOfFile=None):
        global count, flag_homography_to_drawCircle, flag_origin_to_homography, flag_is_YOLO

        ### origin image to homography ###
        if flag_origin_to_homography == 1:
            self.imgInst.imgHomography = self.homographyInst.startHomography(self.imgInst.imgData)
        else:
            self.imgInst.imgHomography = self.imgInst.imgData


        global flag_find_empty_parking_lot
        if flag_find_empty_parking_lot == 1:
            global parkingLot
            self.lineSegInst.emptyParkingLotDetection(frame=imgInst.imgHomography, parkingLotCenter=parkingLot) ################################################################


        ### homography to drawCirble ###
        # not use yolo
        if flag_homography_to_drawCircle == 1 and flag_is_YOLO == 0:
            self.imgInst.redLineSeg = self.lineSegInst.redLineSegmentation(self.imgInst.imgHomography)
            self.imgInst.imgDrawCircle, self.carLocation, self.carAngle = self.lineSegInst.startDrawCircle(frame=self.imgInst.imgHomography.copy(), img_open=self.imgInst.redLineSeg)

            self.imgInst.imgYOLO = self.imgInst.imgDrawCircle

        # use yolo
        elif flag_homography_to_drawCircle == 1 and flag_is_YOLO == 1:
            self.imgInst.imgYOLO, pixelLeftTop_result, pixelRightBottom_result, label_result = self.myDarknetInst.initDetection(self.imgInst.imgHomography.copy())

            # find the car in the cctvView
            car_index = -1
            label_index = 0
            # print('label_result=', label_result, 'pixelLeftTop_result=', pixelLeftTop_result, 'pixelRightBottom_result', pixelRightBottom_result)
            for label in label_result:
                if label == 'cell phone' or label == 'car' or label == 'suitcase' or label == 'skateboard':
                    car_index = label_index
                    break
                else:
                    label_index = label_index + 1

            # using YOLO detection, we would like to easily blackout the background
            # if nothing was detected by YOLO, or there is no car in the image
            if np.shape(label_result)[0] == 0 or car_index == -1:
                # print('yolo did not spot the car')
                self.imgInst.redLineSeg = self.lineSegInst.redLineSegmentation(self.imgInst.imgHomography)

            # there is somthing detected as a car
            else:
                self.imgInst.redLineSeg = self.lineSegInst.redLineSegmentation(self.imgInst.imgHomography,
                                                                               pixelLeftTop=pixelLeftTop_result[car_index],
                                                                               pixelRightBottom=pixelRightBottom_result[car_index])

            cv2.imshow('>> self.imgInst.redLineSeg', self.imgInst.redLineSeg)
            if self.imgInst.redLineSeg is None:
                print('self.imgInst.redLineSeg is None')
            if self.imgInst.imgHomography is None:
                print('self.imgInst.imgHomography is None')

            cv2.waitKey(1)
            # extract car information
            self.imgInst.imgDrawCircle, self.carLocation, self.carAngle = self.lineSegInst.startDrawCircle(frame=self.imgInst.imgHomography.copy(),
                                                                                                           img_open=self.imgInst.redLineSeg)

        else:
            self.imgInst.redLineSeg = self.imgInst.imgHomography
            self.imgInst.imgDrawCircle = self.imgInst.imgHomography
            self.imgInst.imgYOLO = self.imgInst.imgDrawCircle
            self.carLocation = (0, 0)
            self.carAngle = 0

        ### let's draw circles ###
        global flag_drawCircle_to_carView
        if flag_drawCircle_to_carView == 1:

            # if self.flag_first_set_homography_ground_to_carView == 1:
            global flag_1_subscribeImg_2_loadImgFile_3_image2video
            if flag_1_subscribeImg_2_loadImgFile_3_image2video == 1:
                rospy.wait_for_service('remote/carInfo')
                try:
                    carInfo_Request = rospy.ServiceProxy(topic_to_service_in_carInfo, carInfo)
                    getServiceInData = carInfo_Request(0)
                    # print(getServiceInData)
                    self.homography_to_carView = np.reshape(getServiceInData.carViewHomography, newshape=(3, 3))
                    self.carViewShape = (getServiceInData.carViewWidth, getServiceInData.carViewHeight)
                except rospy.ServiceException as e:
                    print('receiving service got error', e)

            elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 2:
                carViewInst = carView_import()
                carViewInst.__init__()
                self.homography_to_carView, _ = carViewInst.setCarViewHomography()
                print('sef.homography_to_carView is ', self.homography_to_carView)
                self.carViewShape = carViewInst.setCarViewShape()

                print('self.homography_to_carView = ', self.homography_to_carView, np.shape(self.homography_to_carView))

                # # do not come here again
                # self.flag_first_set_homography_ground_to_carView = 2

            ## rotation ##
            matrixRotation = cv2.getRotationMatrix2D(center=self.carLocation, angle=360 - self.carAngle, scale=1)
            self.imgInst.imgSeeThrough = cv2.warpAffine(src=self.imgInst.imgDrawCircle, M=matrixRotation, dsize=self.imgInst.imgDrawCircle.shape[:2][::-1])

            # erase the area under the car that makes see-through disturbing
            # print('shape : ', self.imgInst.imgSeeThrough.shape, self.carLocation)
            self.imgInst.imgSeeThrough[self.carLocation[1]:] = np.zeros((self.imgInst.imgSeeThrough.shape[0] - self.carLocation[1],
                                                                         self.imgInst.imgSeeThrough.shape[1],
                                                                         self.imgInst.imgSeeThrough.shape[2]),
                                                                        dtype=np.uint8)

            ## translation ##
            cameraLocation_in_carView = (310, 463)
            tmp = (cameraLocation_in_carView[0] - self.carLocation[0], cameraLocation_in_carView[1] - self.carLocation[1])

            matrixTranslate = np.float32([[1, 0, tmp[0]], [0, 1, tmp[1]]])
            self.imgInst.imgSeeThrough = cv2.warpAffine(src=self.imgInst.imgSeeThrough, M=matrixTranslate, dsize=self.imgInst.imgSeeThrough.shape[:2][::-1])


            if flag_1_subscribeImg_2_loadImgFile_3_image2video == 2:
                global path_carView_undistort_image
                self.carView_undistort = cv2.imread(path_carView_undistort_image + str((100000 + count)) + '.png')
                self.imgInst.imgSeeThrough = cv2.warpPerspective(self.imgInst.imgSeeThrough, self.homography_to_carView, dsize=self.carView_undistort.shape[:2][::-1])
                self.imgInst.imgSeeThrough = cv2.addWeighted(src1=self.imgInst.imgSeeThrough, alpha=0.3, src2=self.carView_undistort, beta=0.7, gamma=0.0, dst=None, dtype=-1)

            elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 1:
                self.imgInst.imgSeeThrough = cv2.warpPerspective(self.imgInst.imgSeeThrough, self.homography_to_carView, dsize=self.carViewShape)#, flags=cv2.WARP_INVERSE_MAP)

        else:
            self.imgInst.imgSeeThrough = self.imgInst.imgDrawCircle

        ## save frames ##
        if flag_saveImg == 1:
            if flag_1_subscribeImg_2_loadImgFile_3_image2video == 1:
                cv2.imwrite(path_save_image + 'cctvView/' + str(100000 + count) + '.png', self.imgInst.imgData)

                # print('save iamge with name : ', path_save_image + 'cctvView_homography/' + str(100000 + count) + '.png')
                # cv2.imwrite((path_save_image + 'cctvView_homography/' + str(100000 + count) + '.png'), self.imgInst.imgHomography)
                #
                # if flag_is_YOLO == 1:
                #     cv2.imwrite((path_save_image + 'cctvView_yolo/' + str(100000 + count) + '.png'), self.imgInst.imgYOLO)
                #
                # if flag_homography_to_drawCircle == 1:
                #     # print('save iamge with name : ', path_save_image + 'cctvView_redLineSeg/' + str(100000 + count) + '.png')
                #     cv2.imwrite((path_save_image + 'cctvView_redLineSeg/' + str(100000 + count) + '.png'), self.imgInst.redLineSeg)
                #     cv2.imwrite((path_save_image + 'cctvView_center/' + str(100000 + count) + '.png'), self.imgInst.imgDrawCircle)
                #
                # if flag_drawCircle_to_carView == 1:
                #     # print('save iamge with name : ', path_save_image + 'cctvView_2_carView/' + nameOfFile[-10:])
                #     cv2.imwrite(path_save_image + 'cctvView_2_carView/' + str(100000 + count) + '.png', self.imgInst.imgSeeThrough)

            elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 2:
                cv2.imwrite(path_save_image + 'cctvView/' + nameOfFile[-10:], self.imgInst.imgData)

                if flag_origin_to_homography == 1:
                    # print('save iamge with name : ', path_save_image + 'cctvView_homography/' + nameOfFile[-10:])
                    cv2.imwrite(path_save_image + 'cctvView_homography/' + nameOfFile[-10:], self.imgInst.imgHomography)

                if flag_is_YOLO == 1:
                    cv2.imwrite(path_save_image + 'cctvView_yolo/' + nameOfFile[-10:], self.imgInst.imgYOLO)
                    # cv2.imshow('yolo', self.imgInst.imgYOLO)
                    # cv2.waitKey(1)

                if flag_homography_to_drawCircle == 1:
                    # print('save iamge with name : ', path_save_image + 'cctvView_redLineSeg/' + nameOfFile[-10:])
                    cv2.imwrite(path_save_image + 'cctvView_redLineSeg/' + nameOfFile[-10:], self.imgInst.redLineSeg)
                    cv2.imwrite(path_save_image + 'cctvView_center/' + nameOfFile[-10:], self.imgInst.imgDrawCircle)

                if flag_drawCircle_to_carView == 1:
                    # print('save iamge with name : ', path_save_image + 'cctvView_2_carView/' + nameOfFile[-10:])
                    cv2.imwrite(path_save_image + 'cctvView_2_carView/' + nameOfFile[-10:], self.imgInst.imgSeeThrough)


        #     # cv2.imshow('self.imgInst.redLineSeg', self.imgInst.redLineSeg)
        # cv2.imshow('self.imgInst.imgHomography', self.imgInst.imgHomography)
        cv2.imshow('drawCircle', self.imgInst.imgDrawCircle)
        #     cv2.imshow('self.imgInst.imgDrawCircle', cv2.resize(np.concatenate((self.imgInst.imgDrawCircle,
        #                                                              self.imgInst.imgYOLO,
        #                                                              np.stack((self.imgInst.redLineSeg,
        #                                                                        self.imgInst.redLineSeg,
        #                                                                        self.imgInst.redLineSeg), axis=2)), axis=1), (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

        # re-publish processed images
        if flag_publishImg == 1:
            try:
                self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(self.imgInst.imgSeeThrough, "bgr8"))

                tmp = Float32MultiArray()
                # tmp.data.clear()
                # tmp.data.push_back(1)
                tmp.layout.dim = [] #MultiArrayDimension()

                # tmp.layout.dim = tuple(['test', 3, 1])
                # tmp.layout.dim.label = str('test')
                # tmp.layout.dim.size = 3
                # tmp.layout.dim.stride = 1

                # tmp.layout.data_offset = 0

                tmp.data = [self.carLocation[0], self.carLocation[1], self.carAngle]
                # tmp.data.length = 3
                self.rospyPubCarInfoArduino.publish(tmp)

                # tmp = carInfoArduino()
                # tmp.carLocationWidth = self.carLocation[0]
                # tmp.carLocationHeight = self.carLocation[1]
                # tmp.carAngle = self.carAngle
                # self.rospyPubCarInfoArduino.publish(tmp)
                
            except CvBridgeError as e:
                print(e)




class Import_cctvView:
    end_of_array = np.zeros(2, dtype=np.float32)

    def __int__(self):
        print('__init__ in the class < Import_cctvView >')
        self.imgInst = ImageClass()
        self.calibInst = Calibration()
        self.dataLoadInst = DataLoadClass(self.imgInst, self.calibInst)

        global cctvLocation3D, carLocationDepth
        self.lineSegInst = LineSegClass(flag_imshow_on=1, flag_print_on=1, cctvLocation3D=cctvLocation3D, carLocationDepth=carLocationDepth)

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
            dataLoadInst.servicepCarInfo()

        # load cctv images in the folders
        else:
            return self.lineSegInst.startDrawCircle(frame=frame_cctv, img_open=self.lineSegInst.redLineSegmentation(frame_cctv))

if __name__ == "__main__":

    print("check the version opencv.")
    print(cv2.__version__)

    # global count
    # global -> error here
    # count = 0
    setParam()

    imgInst = ImageClass()
    calibInst = Calibration()
    homoInst = Homography()
    dataLoadInst = DataLoadClass(imgInst, calibInst, homoInst)

    # global flag_1_subscribeImg_2_loadImgFile_3_image2video
    rospy.init_node('cctvView', anonymous=True)
    try:
        if flag_1_subscribeImg_2_loadImgFile_3_image2video == 1:
            # One python file for one init_node
            if flag_publishImg == 1:
                dataLoadInst.publishImg()
                dataLoadInst.publishCarInfoArduino()
                dataLoadInst.serviceCarInfo()

            dataLoadInst.subscribeImg()
            rospy.spin()

        elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 2:
            if flag_publishImg == 1:
                dataLoadInst.publishImg()
                dataLoadInst.publishCarInfoArduino()
                dataLoadInst.serviceCarInfo()

            dataLoadInst.loadImgInFolder()

        elif flag_1_subscribeImg_2_loadImgFile_3_image2video == 3:
            dataLoadInst.image2video()

    except KeyboardInterrupt:
        print("Shutting down")
