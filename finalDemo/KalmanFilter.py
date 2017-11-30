#!/usr/bin/env python
import numpy as np
import cv2

class KalmanFilter:
    processNoise = np.eye(2, dtype=np.float32)
    measurementNoise = np.eye(2, dtype=np.float32)
    stateTransitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)

    x_post = np.zeros((2, 1), dtype=np.float32)
    p_post = np.zeros(2, dtype=np.float32)
    x_pri = np.zeros((2, 1), dtype=np.float32)
    p_pri = np.eye(2, dtype=np.float32)
    z_prior = 0

    def __init__(self, QfactorProcessNoise, Qfactor):
        # self.processNoise = np.zeros((2,2), dtype=np.float32)
        self.processNoise = self.processNoise * QfactorProcessNoise
        self.measurementNoise = self.measurementNoise * Qfactor

    def initialMeasurement(self, z):
        # print('initial measurement')
        self.x_post = np.array([z, 0], dtype=np.float32).reshape((2, 1))  # [width, d(width)] or [height, d(height)]
        self.p_post = np.eye(2, dtype=np.float32)
        # print('x_post and p_post is')
        # print(self.x_post)
        # print(self.p_post)

        self.x_pri = np.array((z, 0), dtype=np.float32).reshape((2, 1))
        self.p_pri = np.eye(2, dtype=np.float32)
        # print('x_pri and p_pri is')
        # print(self.x_pri)
        # print(self.p_pri)

        # this value is to save the prior position(x_k-1) to get the current velocity(v_k)
        self.z_prior = z

    def timeUpdate(self):
        # print('time update')
        self.x_pri = np.matmul(self.stateTransitionMatrix, self.x_post)
        self.p_pri = np.matmul(np.matmul(self.stateTransitionMatrix, self.p_post), self.stateTransitionMatrix.transpose()) + self.processNoise
        # print('x_pri and p_pri is')
        # print(self.x_pri)
        # print(self.p_pri)

    def measurementUpdate(self, z):
        # print('measurement update')
        z_2by1 = np.array((z, z - self.z_prior), dtype=np.float32).reshape((2, 1))  # [position, velocity]

        # this value is to save the prior position(x_k-1) to get the current velocity(v_k)
        self.z_prior = z

        self.kalmanGain = np.matmul(self.p_pri, np.linalg.inv(self.p_pri + self.measurementNoise))
        self.x_post = self.x_pri + np.matmul(self.kalmanGain, z_2by1 - self.x_pri)  # [width, d(width)] or [height, d(height)]
        self.p_post = (np.eye(2, dtype=np.float32) - self.kalmanGain) * self.p_pri

        # print('x_post and p_post is')
        # print(self.x_post)
        # print(self.p_post)

        self.timeUpdate()