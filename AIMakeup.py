# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:36:11 2017

@author: Quantum Liu
"""

import sys,os,traceback
import cv2
import dlib
import numpy as np

def get_rect(im,landmarks):
    return im[landmarks[:,1],landmarks[:,0],:]

class NoFace(Exception):
    '''
    没脸
    '''
    pass

class Organ():
    '''
    五官部位类
    '''
    def __init__(self,im_bgr,name):
        self.im_bgr,self.name=im_bgr,name
        
class Face():
    '''
    脸类
    '''
    def __init__(self,im_face,lanmarks,rect,index):
        self.im_face,self.lanmarks,self.rect,self.index=im_face,lanmarks.copy(),rect,index
        
        #五官等标记点
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))

        # 人脸的完整标记点
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                                       self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)

class Makeup():
    '''
    化妆器
    '''
    def __init__(self,predictor_path="./data/shape_predictor_68_face_landmarks.dat"):
        self.photo_path=[]
        self.PREDICTOR_PATH = predictor_path
        self.faces={}
        
        #人脸定位、特征提取器，来自dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

    def get_landmarks(self,im,fname,n=1):
        '''
        人脸定位和特征提取，定位到两张及以上脸或者没有人脸将抛出异常
        im:
            照片的numpy数组
        fname:
            照片名字的字符串
        返回值:
            人脸特征(x,y)坐标的矩阵
        '''
        rects = self.detector(im, 1)
        
        if len(rects) <1:
            raise NoFace('Too many faces in '+fname)
        return {fname:[Face(*(lambda im,landmarks:(get_rect(im,landmarks),landmarks))(im,np.array([[p.x, p.y] for p in self.predictor(im, rect).parts()])),rect,i) for i,rect in enumerate(rects)]}

    def read_im(self,fname,scale=1):
        '''
        读取图片
        '''
# =============================================================================
#         im = cv2.imread(fname, cv2.IMREAD_COLOR)
# =============================================================================
        im = cv2.imdecode(np.fromfile(fname,dtype=np.uint8),-1)
        if type(im)==type(None):
            print(fname)
            raise ValueError('Opencv read image {} error, got None'.format(fname))
        return im

    def read_and_mark(self,fname):
        im=self.read_im(fname)
        return im,self.get_landmarks(im,fname)

    def light(self,pos,delta):
        pass
    