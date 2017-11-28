#!/usr/bin/env python

import rospy
import roslib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from yolo.srv import *

import cv2
import numpy as np
import glob
from random import randint

ROS_TOPIC = '/desktop/image_color'
flag_publishImg = 1

inputSize = (416, 416)
count = 0
param_cfg = 'cfg/yolo.cfg'
param_model = 'yolo.weights'
param_class_name = 'data/coco.names'
min_confidence = 0.3

path_load_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView/*.png'
path_save_image = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView_yolo/'
path_darknet = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/darknet/'

# params = "{ help           | false | print usage         }"
#       "{ cfg            |  /home/jaesungchoe/catkin_ws/src/yolo/src/darknet/cfg/yolo.cfg     | model configuration }"
#       "{ model          |  /home/jaesungchoe/catkin_ws/src/yolo/src/darknet/yolo.weights     | model weights       }"
#       "{ camera_device  | 0     | camera device number}"
#       "{ video          |  /home/jaesungchoe/catkin_ws/src/futureCarCapstone/src/see-through-parking-using-augmented-reality/finalDemo/fisheye_lens/cctvView_yolo/cctvView_yolo.avi     | video or image for detection}"
#       "{ min_confidence | 0.24  | min confidence      }"
#       "{ class_names    |  /home/jaesungchoe/catkin_ws/src/yolo/src/darknet/data/coco.names     | class names         }";


class MyDarknet:
    def __int__(self):
        print('init darknet')
        # print('what the fuck : ', path_darknet + param_cfg)
        self.net = cv2.dnn.readNetFromDarknet((path_darknet + param_cfg), (path_darknet + param_model))
        # print('net is ', net)
        self.setLabel()
        # print('shape of self.label is ', np.shape(self.label))
        # print('self.label[0] is ', self.label[0])

    def setLabel(self):
        f = open((path_darknet + param_class_name), 'r')
        self.label = []
        for line in f:
            self.label.append(line)

        self.label = np.array(self.label)

    def initDetection(self, frame):
        img = cv2.resize(frame, inputSize)
        inputBlob = cv2.dnn.blobFromImage(img, 1.0/255.0)

        self.net.setInput(inputBlob, 'data')
        detectionMat = self.net.forward('detection_out')

        # print(detectionMat)
        # self.parseFPS()D
        return self.parseDetection(result=detectionMat, img=frame)

    def parseFPS(self):
        freq = cv2.getTickFrequency() / 1000
        time = self.net.getPerProfile() / freq
        self.fps = 1000 / time
        print('fps is ', self.fps)

    def parseDetection(self, result, img):
        # print('result.shape is ', np.shape(result))
        pixelLeftTop_result = []
        pixelRightBottom_result = []
        label_result = []

        for i in range(result.shape[0]):
            probability_index = 5
            probability_size = result.shape[1] - probability_index
            # print('result[0][:5] = ', result[0][:5])
            objectClass = np.argmax(result[i][probability_index:])
            # print('objectClass is ', objectClass)
            confidence = result[i][probability_index + objectClass]

            global min_confidence
            if confidence > min_confidence:
                (x, y, width, height) = result[i][0:3+1]
                # print('For label : ', i, 'x, y, width, height, image.shape', x, y, width, height, img.shape)
                # print('x * img.shape[1], y * img.shape[0] = ', x*img.shape[1], y*img.shape[0])
                xLeftTop = (x - (width / 2.0)) * img.shape[1]
                yLeftTop = (y - (height / 2.0)) * img.shape[0]
                xRightBottom = (x + (width / 2.0)) * img.shape[1]
                yRighttop = (y + (height / 2.0)) * img.shape[0]
                # print('xLeftTop, yLeftTop, xRightBottom, yRighttop  : ', xLeftTop, yLeftTop, xRightBottom, yRighttop)

                if objectClass < self.label.shape[0]:
                    label_to_put = self.label[objectClass] + str(confidence)
                    color_to_print = (randint(100, 150), randint(150,200), randint(150,255)) #(255,0,255)
                    pixelLeftTop = (int(xLeftTop), int(yLeftTop))
                    pixelRightBottom = (int(xRightBottom), int(yRighttop))
                    cv2.rectangle(img=img, pt1=pixelLeftTop, pt2=pixelRightBottom, color=color_to_print, thickness=2)
                    cv2.putText(img=img, text=label_to_put, org=pixelLeftTop,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color_to_print, thickness=2)

                    pixelRightBottom_result.append(pixelRightBottom)
                    pixelLeftTop_result.append(pixelLeftTop)
                    label_result.append(self.label[objectClass])
                    # cv2.circle(img=img, center=(int(x*img.shape[1]), int(y*img.shape[0])), radius=5, color=(0,0,255), thickness=2)
                    # cv2.circle(img=img, center=(int(width*img.shape[1]), int(height*img.shape[0])), radius=5, color=(0, 0, 255), thickness=2)

                # for indicating_the_car in ['cell phone', 'car']:
                #     if self.label[objectClass] == indicating_the_car:
                #         pixelLeftTop_result = pixelLeftTop
                #         pixelRightBottom_result = pixelRightBottom
                #     elif i == (result.shape[0] - 1):
                #         pixelLeftTop_result = pixelLeftTop
                #         pixelRightBottom_result = pixelRightBottom

        # cv2.imshow('result of yolo', img)
        # cv2.waitKey(1)
        return img, pixelLeftTop_result, pixelRightBottom_result, label_result #'list' object has no attribute 'copy'

class DataLoadClass:
    imgInst = []
    calibInst = []
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1

    def __init__(self, imgInst, myDarknetInst):
        self.imgInst = imgInst
        self.myDarknetInst = myDarknetInst
        self.bridge = CvBridge()

    def subscribeImg(self):
        # print('start to subscribe image')
        rospy.init_node('MyDarknet', anonymous=True)
        self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
        # automatically go to the callback function : self.callback()

    def loadImgInFolder(self):
        global fileList
        fileList = glob.glob(path_load_image)
        print('path_load_image is ', path_load_image)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        print('what is fileList', fileList, '\n')

        global count
        fileList = np.sort(fileList)
        for i in fileList:
            count = count + 1
            # print('The ', count, '-th image is ', i)
            self.imgInst.saveImage(cv2.imread(i))
            frameAnnotated = self.myDarknetInst.initDetection(self.imgInst.imgData)

            cv2.imwrite(path_save_image + i[-9:], self.imgInst.imgData)

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
        self.rospyPubImg = rospy.Publisher('MyDarknet/image_annotated', Image, queue_size=10)
        rospy.Rate(10)  # 10Hz

    def serviceCarInfo(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish yoloResult')
        self.rospyServiceCarInfo = rospy.Service('MyDarknet/yoloResult', yoloResult, self.yoloResult_Request)

    def yoloResult_Request(self, req):
        print('yoloResult_Request : ', (self.pixelLeftTop[0], self.pixelLeftTop[1], self.pixelRightBottom[0], self.pixelRightBottom[1]))
        return yoloResultResponse(self.pixelLeftTop[0], self.pixelLeftTop[1], self.pixelRightBottom[0], self.pixelRightBottom[1])

    def callback(self, data):
        global count
        try:
            count = count + 1
            # print('count =', count)
            frameAnnotated, self.pixelLeftTop, self.pixelRightBottom = self.myDarknetInst.initDetection(self.bridge.imgmsg_to_cv2(data, "bgr8"))
            # cv2.imshow('frameAnnotated', frameAnnotated)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

        if flag_publishImg == 1:
            try:
                self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(frameAnnotated, "bgr8"))
            except CvBridgeError as e:
                print(e)

class ImgClass:
    height = 0
    width = 0
    imgData = None
    imgAnnotated = None

    def saveImage(self, img):
        self.imgData = img
        self.height, self.width = self.imgData.shape[:2]

if __name__ == "__main__":
    print("check the version opencv.")
    print(cv2.__version__)
    print(cv2.__file__)

    myDarknetInst = MyDarknet()
    myDarknetInst.__int__() # do I have to write this initializer??
    imgInst = ImgClass()

    dataLoadInst = DataLoadClass(imgInst=imgInst, myDarknetInst=myDarknetInst)
    dataLoadInst.loadImgInFolder()
    # dataLoadInst.subscribeImg()
    # dataLoadInst.publishImg()
    #  rospy.spin()



