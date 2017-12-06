#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge, CvBridgeError
import message_filters

import rospy
import roslib

import glob
import os

import numpy as np
import cv2
import sys

import time
from ImageClass import *
from ViewTransform import Homography


#https://github.com/kushalvyas/Python-Multiple-Image-Stitching/blob/master/code/pano.py
import cv2
import numpy as np

#flag
flag_subscribe_new_image_not_load_old_image = 1
flag_publish_img = 1

flag_load_calibrated_result = 1
flag_load_detected_result = 0
flag_fisheye_calibrate = 1
flag_save_image_onlyWhichDetectCheckeboard = 0
flag_show_image = 1

cutDownAllHeight = 490 #####################################################################################
shape_imgHomography = (620, 800)
#(550, 480) #partially erased
#(610, 565) is the all parking lot, 510 is only with visible area

srcPixel_Img_cctvView = np.array([
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

dstPixel_ground_cctvView = np.array([

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

srcPixel_Img_startView = np.array([
    #starting cctv
    [130, 186], [913, 357],
    [135, 576], [401, 556], [486, 557], [576, 552], [663, 544], [751, 542]
])

dstPixel_ground_startView = np.array([
    #starting CCTV
    [323, 405], [616, 452],
    [323, 565], [436, 565], [466, 565], [496, 565], [526, 565], [556, 565]
])


#paramter
fileList = []
count = 0



class dataLoadType:

    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1
    flag_first_didHomography = 1

    def __init__(self, imgInst_cctvView, imgInst_startView):
        self.imgInst_cctvView = imgInst_cctvView
        self.imgInst_startView = imgInst_startView
        self.bridge = CvBridge()
        self.homoInst_cctvView = Homography()
        self.homoInst_startView = Homography()

    def subscribeImage(self):
        print('start to subscribe image')
        #rospy.init_node('dataLoadType', anonymous=True)
        self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
        #automatically go to the callback function : self.callback()

    def subscribeMultiImage_Sync(self):
        print('start to subscribe multi iamges')
        subImg_JS = message_filters.Subscriber('desktop/image_annotated', Image)
        subImg_Francois = message_filters.Subscriber('remote/image_annotated', Image)
        # subImg_JS = message_filters.Subscriber('jaesung_lens_camera/image_color', Image)
        # subImg_Francois = message_filters.Subscriber('francois_lens_camera/image_color', Image)
        tss = message_filters.TimeSynchronizer([subImg_JS, subImg_Francois], 1000)
        tss.registerCallback(self.callback)

    def subscribeMultiImage_Async(self):
        print('start to subscribe Multi images asynchronously')
        self.rospySubImg_cctvView = message_filters.Subscriber('desktop/image_color/compressed', CompressedImage)
        self.rospySubImg_startView = message_filters.Subscriber('remote/image_color/compressed', CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer([self.rospySubImg_cctvView, self.rospySubImg_startView], 10, 5)
        ts.registerCallback(self.callback)

    def loadImageInFiles(self):
        global fileList
        fileList = glob.glob(path_image_database)
        fileList = sorted(fileList)
        print('path_image_database is ', path_image_database)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        print('what is fileList', fileList, 'num is ', num_of_image_in_database)

        global count
        count = 0
        for i in range(0, num_of_image_in_database, 2):
            count = count + 1
            print('The ', i, '-th image is under processing', 'count is ', count, 'and ', fileList[i])
            self.imgInst_cctvView.saveImage(cv2.imread(fileList[i]))
            self.imgInst_startView.saveImage(cv2.imread(fileList[i+1]))
            self.wrapper()

    def publishImage(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish JS image')
        self.rospyPubImg = rospy.Publisher('cctvOverView/image_color', Image, queue_size=10)
        #rospy.init_node('calibrated_JS_lens_fisheye', anonymous=True)
        # rate = rospy.Rate(10)  # 10Hz

    def callback(self, img_encode_cctvView=None, img_encode_startView=None):
        global count
        try:
            print('>> into callback function')
            # parse message into image
            # bgr8: CV_8UC3, color image with blue-green-red color order and 8b

            np_arr = np.fromstring(img_encode_cctvView.data, np.uint8)
            self.imgInst_cctvView.saveImage(cv2.imdecode(np_arr, cv2.IMREAD_COLOR))  # cv2.CV_LOAD_IMAGE_COLOR is out of version
            # self.imgInst_cctvView.saveImage(self.bridge.imgmsg_to_cv2(img_cctvView, "bgr8"))

            np_arr_ = np.fromstring(img_encode_startView.data, np.uint8)
            self.imgInst_startView.saveImage(cv2.imdecode(np_arr_, cv2.IMREAD_COLOR))  # cv2.CV_LOAD_IMAGE_COLOR is out of version
            # self.imgInst_startView.saveImage(self.bridge.imgmsg_to_cv2(img_startView, "bgr8"))

            ## homography
            global shape_imgHomography
            self.homoInst_cctvView.setHomogrphay(srcPixel_Img_cctvView, dstPixel_ground_cctvView, outputShape=shape_imgHomography)
            self.homoInst_startView.setHomogrphay(srcPixel_Img_startView, dstPixel_ground_startView, outputShape=shape_imgHomography)

            self.imgInst_cctvView.imgHomography = self.homoInst_cctvView.startHomography(frame=self.imgInst_cctvView.imgData)
            self.imgInst_startView.imgHomography = self.homoInst_startView.startHomography(frame=self.imgInst_startView.imgData)

            #stitch two images
            global cutDownAllHeight
            # shape = np.array(self.imgInst_cctvView.imgHomography.shape) #shape = [height, width, channel]
            # shape[0] = shape[0] - cutDownAllHeight
            # self.imgInst_cctvView.imgHomography[cutDownAllHeight:, :, :] = np.zeros(tuple(shape), dtype=np.uint8)
            # self.imgInst_startView.imgHomography[:cutDownAllHeight, :, :] = np.zeros((cutDownAllHeight, shape[1], shape[2]), dtype=np.uint8)
            self.imgInst_cctvView.imgHomography[cutDownAllHeight:, :, :] = self.imgInst_startView.imgHomography[cutDownAllHeight:, :, :]

            cv2.imshow('self.imgInst_cctvView.imgHomography', self.imgInst_cctvView.imgHomography)
            # cv2.imshow('self.imgInst_startView.imgHomography', self.imgInst_startView.imgHomography)
            cv2.waitKey(1)

            # if you want to work asynchronously, edit the lines below
            count = count + 1
            
            # # alpha = 0.2
            # # tmp1 = cv2.addWeighted(src1=self.imgInst_startView.imgData, alpha=alpha, src2=self.imgInst_cctvView.imgData, beta=(1 - alpha), gamma=0.0, dst=None, dtype=-1)
            # alpha = 0.5
            # tmp2 = cv2.addWeighted(src1=self.imgInst_startView.imgHomography, alpha=alpha,
            #                        src2=self.imgInst_cctvView.imgHomography, beta=(1 - alpha), gamma=0.0, dst=None, dtype=-1)
            # # # alpha = 0.75
            # # # tmp3 = cv2.addWeighted(src1=self.imgInst_startView.imgData, alpha=alpha, src2=self.imgInst_cctvView.imgData, beta=(1 - alpha), gamma=0.0, dst=None, dtype=-1)
            # #
            # # # cv2.imshow('See-Through frames', np.concatenate((tmp1, tmp2, tmp3), axis=1))
            # cv2.imshow('See-Through frames', tmp2)
            # cv2.waitKey(1)
            #
            # if flag_publish_img == 1:
            #     try:
            #         self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(tmp2, "bgr8"))
            #         cv2.waitKey(1)
            #     except CvBridgeError as e:
            #         print(e)



        except CvBridgeError as e:
            print(e)

    
if __name__ == '__main__':

    print("check the version opencv.")
    print(cv2.__version__)

    imgInst_cctvView = ImageClass()
    imgInst_startView = ImageClass()

    dataLoadType_inst = dataLoadType(imgInst_cctvView, imgInst_startView)

    rospy.init_node('imageStitching', anonymous=True)
    try:
        if flag_subscribe_new_image_not_load_old_image == 1:
            dataLoadType_inst.subscribeMultiImage_Async()
            # dataLoadType_inst.subscribeMultiImage_Sync()
            

            if flag_publish_img == 1:
                dataLoadType_inst.publishImage()

            rospy.spin()
        else:
            dataLoadType_inst.loadImageInFiles()
             
    except KeyboardInterrupt:
        print('end')


