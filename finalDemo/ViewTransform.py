#!/usr/bin/env python
import numpy as np
import cv2
import glob
from cctvView import Import_cctvView
from math import pi

# CarViewHomography_2_CctvViewHomography:
class ViewTransform:
    flag_1_subscribeImg_2_loadImgFile = 0
    count = -1
    cameraLocation_in_carView = (310, 463)

    def __init__(self, flag, cctvViewPath=None, dataLoadInst=None):
        print('in the __init__ of ViewTransforms')
        self.flag_1_subscribeImg_2_loadImgFile = flag

        # subscribe cctv frames
        if self.flag_1_subscribeImg_2_loadImgFile == 1:
            self.dataLoadInst = dataLoadInst
            self.import_cctvView_inst = Import_cctvView()
            self.import_cctvView_inst.__int__()

        # load cctv frames
        elif self.flag_1_subscribeImg_2_loadImgFile == 2: #flag_1_subscribeImg_2_loadImgFile == 2
            self.fileList_cctv = glob.glob(cctvViewPath)
            self.fileList_cctv = np.sort(self.fileList_cctv)
            # print('>>>>> self.fileList_cctv', self.fileList_cctv)
            # print('>> is the class Import_cctvView initialized?')
            self.import_cctvView_inst = Import_cctvView()
            # print('>> is the class Import_cctvView initialized? __init__ is needed ?') #yes
            self.import_cctvView_inst.__int__()

    def getCarLocationAngle(self):
        # print('find location and angle of the car in cctvView')
        if self.flag_1_subscribeImg_2_loadImgFile == 2: #
            # location = np.zeros(2, dtype=np.float32)
            self.count = self.count + 1
            return self.import_cctvView_inst.getCarLocationAngle(frame_cctv=cv2.imread(self.fileList_cctv[self.count]))  # []##########################################################

        elif self.flag_1_subscribeImg_2_loadImgFile == 1:
            # print('dataLoadInst.serviceCarInfo()', dataLoadInst.serviceCarInfo(), 'count is ', count)
            return self.dataLoadInst.serviceInCarInfo()

    def generate_car_roi_in_cctvView(self, frameCarView, shape_imgHomography, emptyFrame=None):
        # why it is error???
        # carLocation = np.zeros(2, dtype=np.int64)
        # frameCCTVView, (carLocation[0], carLocation[1]), carAngle = self.getCarLocationAngle() #carLocation = (width, height), carAngle=(theta) refer to cctvView.py
        if self.flag_1_subscribeImg_2_loadImgFile == 1:
            # if None .........................it encounters errors 'NoneType' object is not iterable #############################################################################################

            # bad idea. because carLocation need to become tuple
            # carLocation = np.zeros(2, dtype=np.int32)
            carLocation, carAngle = self.getCarLocationAngle()  # carLocation = (width, height), carAngle=(theta) refer to cctvView.py
            frameCCTVView = None
        elif self.flag_1_subscribeImg_2_loadImgFile == 2:
            frameCCTVView, carLocation, carAngle = self.getCarLocationAngle()  # carLocation = (width, height), carAngle=(theta) refer to cctvView.py


        tmp = (carLocation[0] - self.cameraLocation_in_carView[0], carLocation[1] - self.cameraLocation_in_carView[1])
        print('tep = ', tmp, 'carLocation = ', carLocation, 'cameraLocation_incarView = ', self.cameraLocation_in_carView)
        # matrixTranslate = np.float32([[1, 0, tmp[0]], [0, 1, tmp[1]]])
        # matrixRotation = cv2.getRotationMatrix2D(center=carLocation, angle=carAngle, scale=1)  # carLocation

        # merge the matrix makes seeThorugh less accurate
        # myEmptyFrame = frameCarView.copy()
        myMatrix = cv2.getRotationMatrix2D(center=carLocation, angle=carAngle, scale=1)
        myMatrix[0][2] = myMatrix[0][2] + tmp[0]
        myMatrix[1][2] = myMatrix[1][2] + tmp[1]
        emptyFrame = cv2.warpAffine(src=frameCarView, M=myMatrix, dsize=shape_imgHomography)

        # emptyFrame = cv2.warpAffine(src=frameCarView, M=matrixTranslate, dsize=shape_imgHomography)
        # emptyFrame = cv2.warpAffine(src=emptyFrame, M=matrixRotation, dsize=shape_imgHomography)

        # draw cameraLocation_in_cctvView
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

class Homography:
    homography = np.zeros((3, 3), dtype=np.float32)  ##############################################################################
    warpResultShape = np.zeros(2, dtype=np.uint16)

    def __init__(self):
        print('initiate the class <Homography>')

    def setHomogrphay(self, srcPixel_image, dstPixel_ground, outputShape, shift=None):
        if shift is not None:
            for i in range(dstPixel_ground.shape[0]):
                dstPixel_ground[i][0] = dstPixel_ground[i][0] + shift[0]
                dstPixel_ground[i][1] = dstPixel_ground[i][1] + shift[1]

        self.homography, mask = cv2.findHomography(srcPoints=srcPixel_image,
                                                   dstPoints=dstPixel_ground,
                                                   method=0)#cv2.RANSAC, maxIters=100)
        self.warpResultShape = outputShape
        # print('homography is ', self.homography, 'np.dtype( ... ) = ', np.dtype(self.homography[0][0])) ###############################################
        # print('shape of the mask is', np.shape(mask))

    def startHomography(self, frame, warpflags=0):
        return cv2.warpPerspective(frame, self.homography, self.warpResultShape, flags=(warpflags + cv2.INTER_LINEAR))#(360, 300 + 180) #cv2.WARP_INVERSE_MAP