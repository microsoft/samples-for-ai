#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:25:45 2018

@author: ubuntu
"""

import random
import cv2
import numpy as np

class DatasetFetch():
    def __init__(self, commonParams, datasetParams):
        '''
        Args:
            commonParams: dict
            datasetParams: dict
        '''
        self.dataPath = str(datasetParams['path'])
        self.width = int(commonParams['image_size'])
        self.height = int(commonParams['image_size'])
        self.batchSize = int(commonParams['batch_size'])
        self.threadNum = int(datasetParams['thread_num'])
        self.maxObjects = int(commonParams['max_objects_per_image'])
        
        self.recordList = []
        
        inputFile = open(self.dataPath, 'r')
        
        for line in inputFile:
            line = line.strip()
            ss = line.split(' ')
            ss[1:] = [float(num) for num in ss[1:]]
            self.recordList.append(ss)
            
    def recordProcess(self, record):
        '''
        Record image info. -- path object coordinates.
        
        Args:
            record: imagePath xmin1 ymin1 xmax1 ymax1 calss1 xmin2 ymin2 xmax2 ymax2 class2
        
        Returns:
            image: 3-D ndarray
            labels: 2-D list [self.maxObjects, 5] --> [xCenter, yCenter, w, h, classNum]
            objectNum: int of total object number
        '''
        image = cv2.imread(record[0])
        if image == None:
            raise Exception("Couldn't load image!")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        
        widthRate = self.width*1.0 / w
        heightRate = self.height*1.0 / h
        
        image = cv2.resize(image, (self.height, self.width))
        
        labels = [[0, 0, 0, 0, 0]] * self.maxObjects
        i = 1
        objectNum = 0
        
        while i < len(record):
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            classNum = record[i + 4]
            #Real position of ficed size
            xcenter = (xmin + xmax) * 1.0 / 2 * widthRate
            ycenter = (ymin + ymax) * 1.0 / 2 * heightRate
            
            boxW = (xmax - xmin) * widthRate
            boxH = (ymax - ymin) * heightRate
            
            labels[objectNum] = [xcenter, ycenter, boxW, boxH, classNum]
            objectNum += 1
            
            i += 5
            if objectNum >= self.maxObjects:
                break
        
        return [image, labels, objectNum]
    
    def batch(self):
        '''
        Get batch.
        
        Returns:
            images: 4-D ndarray [batch size, h, w, 3]
            labels: 3-D ndarray [batch size, max objects, 5]
            objectsNum: 1-D ndarray [batch size]
        '''
        images = []
        labels = []
        objectsNum = []
        
        for i in range(self.batchSize):
            image, label, objectNum = self.imageLabelQueue.get()
            images.append(image)
            labels.append(label)
            objectsNum.append(objectNum)
            
        images = np.asarray(images, dtype=np.float32)
        images = images/255*2 - 1
        labels = np.asarray(labels, dtype = np.float32)
        objectsNum = np.asarray(objectsNum, dtype = np.float32)
        return images, labels, objectsNum