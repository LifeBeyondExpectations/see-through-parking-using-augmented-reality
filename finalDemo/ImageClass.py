#!/usr/bin/env python
import numpy as np
import cv2

class ImageClass:
    height = 0
    width = 0
    imgData = None
    imgUndistort = None
    imgHomography = None
    imgWarp = None
    imgYOLO = None
    imgRedLineSeg = None
    imgDrawCircle = None
    imgSeeThrough = None

    def saveImage(self, img, resize=None):
        #resize = (width, height)
        #img.shape = (height, width, channel)
        if resize is not None:
            self.imgData = cv2.resize(src=img, dsize=resize)
        else:
            self.imgData = img.copy()

        self.width, self.height = self.imgData.shape[:2]