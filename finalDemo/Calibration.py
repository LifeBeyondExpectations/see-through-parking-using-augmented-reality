#!/usr/bin/env python
import numpy as np
import cv2
import pickle

class Calibration:
    camera_matrix = np.zeros((3,3), dtype=np.float32)
    new_camera_matrix = np.zeros((3,3), dtype=np.float32)
    distCoeffs = np.zeros(4, dtype=np.float)
    height_trainImage = 0
    width_trainImage = 0


    def __int__(self, path_pickle):
        print('initialize the class <Calibration>')
        self.path_pickle_calibration_variable = path_pickle

    def loadVarAfterCalibration(self):
        path_pickle_calibration_variable = '/home/jaesungchoe/catkin_ws/src/future_car_capstone/src/see-through-parking-using-augmented-reality/finalDemo/calib_result_JS_fisheye.pickle'
        with open(path_pickle_calibration_variable) as f:
            self.camera_matrix, self.distCoeffs, self.new_camera_matrix, self.width_trainImage, self.height_trainImage, _, _, _ = pickle.load(f)

            print('camera matrix is ')
            print(self.camera_matrix)
            print('new camera Matrix is ')
            print(self.new_camera_matrix)
            print('distort Coeffs is ')
            print(self.distCoeffs)
            print('width, height is ', self.width_trainImage, self.height_trainImage)

    def startUndistort(self, frame):
        return cv2.fisheye.undistortImage(frame, self.camera_matrix, self.distCoeffs,
                                                       Knew=self.camera_matrix, new_size=frame.shape[:2][::-1])#(1288, 964)
