#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:12:15 2018

@author: root
"""
import tensorflow as tf
import numpy as np
import re

from yolo.net.net import Net

class YoloTinyNet(Net):
    '''
    Yolo net implementation.
    '''
    def __init__(self, commonParams, netParams, test = False):
        '''
        Init the object.

        Args:
            commonParams: a dict of pretrained parameters
            netParams: a dict of trainable parameters
        '''
        super(YoloTinyNet, self).__init__(commonParams, netParams)

        #Process the parameters.
        self.imageSize = int(commonParams['image_size'])
        self.numClasses = int(commonParams['num_classes'])
        self.cellSize = int(netParams['cell_size'])
        self.boxesPerCell = int(netParams['boxes_per_cell'])
        self.batchSize = int(commonParams['batch_size'])
        self.wd = float(netParams['weight_decay'])

        if not test:
            self.objectScale = float(netParams['object_scale'])
            self.noobjectScale = float(netParams['noobject_scale'])
            self.classScale = float(netParams['class_scale'])
            self.coordScale = float(netParams['coord_scale'])

    def yoloTinyModel(self, images):
        '''
        Build your yolo model.

        Args:
            images: 4-D tensor [batch size, height, width, channels]

        Return:
            predicts: [batch size, cell size, cell size, #class + 5*boxes per cell]
        '''
        conv_num = 1

        temp_conv = self.conv2d('conv' + str(conv_num), images, [3, 3, 3, 16], stride=1)
        conv_num += 1

        temp_pool = self.maxPool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 16, 32], stride=1)
        conv_num += 1

        temp_pool = self.maxPool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 32, 64], stride=1)
        conv_num += 1

        temp_conv = self.maxPool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 128], stride=1)
        conv_num += 1

        temp_conv = self.maxPool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
        conv_num += 1

        temp_conv = self.maxPool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
        conv_num += 1

        temp_conv = self.maxPool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1

        temp_conv = tf.transpose(temp_conv, (0, 3, 1, 2))

        #Fully connected layer
        local1 = self.fullyConnectLayer('local1', temp_conv, self.cellSize * self.cellSize * 1024, 256)

        local2 = self.fullyConnectLayer('local2', local1, 256, 4096)

        local3 = self.fullyConnectLayer('local3', 
                                        local2, 
                                        4096, 
                                        self.cellSize * self.cellSize * (self.numClasses + self.boxesPerCell * 5), 
                                        leaky=False, 
                                        pretrain=False, 
                                        train=True)

        n1 = self.cellSize * self.cellSize * self.numClasses

        n2 = n1 + self.cellSize * self.cellSize * self.boxesPerCell

        class_probs = tf.reshape(local3[:, 0:n1], (-1, self.cellSize, self.cellSize, self.numClasses))
        scales = tf.reshape(local3[:, n1:n2], (-1, self.cellSize, self.cellSize, self.boxesPerCell))
        boxes = tf.reshape(local3[:, n2:], (-1, self.cellSize, self.cellSize, self.boxesPerCell * 4))

        local3 = tf.concat([class_probs, scales, boxes], 3)

        predicts = local3

        return predicts

    def iou(self, boxes1, boxes2):
        '''
        Compute IOU.

        Args:
            boxes1: 4-D tensor [cell size, cell size, boxes per cell, 4] 4 means [x center, y center, width, height]
            boxes2: 1-D tensor [4] 4 means [x center, y center, width, height]

        Return:
            iou: 3-D tensor [cell size, cell size, boses per cell]
        '''
        #Calculate the left-up and right-bottom coordinatino of boses1 & 2.
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2]/2,
                           boxes1[:, :, :, 1] - boxes1[:, :, :, 3]/2,
                           boxes1[:, :, :, 0] + boxes1[:, :, :, 2]/2,
                           boxes1[:, :, :, 1] + boxes1[:, :, :, 3]/2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

        boxes2 = tf.stack([boxes2[0] - boxes2[2]/2,
                           boxes2[1] - boxes2[3]/2,
                           boxes2[0] + boxes2[2]/2,
                           boxes2[1] + boxes2[3]/2])

        # left up intersection point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # the variable intersection contains the width and height of the intersection area.
        intersection = lu - rd

        interSquare = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        #Make sure there is intersection
        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

        interSquare = interSquare * mask

        s1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        s2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return interSquare/(s1 + s2 - interSquare + 1e-6)


    def cond1(self, num, objectNum, loss, predict, labels, nilboy):
        '''
        '''
        return num < objectNum

    def body1(self, num, objectNum, loss, predict, labels, nilboy):
        '''
        Calculate loss.

        Args:
            num: spedify which image is to be processed
            objectNum: #objects in an image
            loss: [class loss, object loss, no object loss, coord loss]
            predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
            labels: [max_objects, 5]  (x_center, y_center, w, h, class)
                --- > class and coord
                --- > x_center is the x value of resized image. the same to y_center
            nilboy: has/no objects
        '''
        #Get label form labels by the varibale num
        label = labels[num]
        label = tf.reshape(label, [-1])

        minX = (label[0] - label[2] / 2) / (self.imageSize / self.cellSize)
        maxX = (label[0] + label[2] / 2) / (self.imageSize / self.cellSize)
        minY = (label[1] - label[3] / 2) / (self.imageSize / self.cellSize)
        maxY = (label[1] + label[3] / 2) / (self.imageSize / self.cellSize)

        #Determine which cell is the object belongs to.
        minX = tf.floor(minX)
        minY = tf.floor(minY)
        maxX = tf.ceil(maxX)
        maxY = tf.ceil(maxY)

        #temp: if a cell contains an object, temp = 1, else 0
        temp = tf.cast(tf.stack([maxY - minY, maxX - minX]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)

        #temp: if a cell doesn't contains an object, temp = 0
        #Which means pad it to S*S scale.
        temp = tf.cast(tf.stack([minY, self.cellSize - maxY, minX, self.cellSize - maxX]),
                       dtype=tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, 'CONSTANT')

        #Calculate which cell contains the center point of the object.
        centerX = label[0] / (self.imageSize / self.cellSize)
        centerX = tf.floor(centerX)
        centerY = label[1] / (self.imageSize / self.cellSize)
        centerY = tf.floor(centerY)
        response = tf.ones([1, 1], tf.float32)

        # pad to S*S scale.
        temp = tf.cast(tf.stack([centerY, self.cellSize - centerY - 1, centerX,self.cellSize - centerX -1]),
                       dtype=tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, 'CONSTANT')

        #predictBoxes: predicted boxes
        predictBoxes = predict[:, :, self.numClasses + self.boxesPerCell:]

        # 7 * 7 * 2 * 4
        predictBoxes = tf.reshape(predictBoxes,
                                  [self.cellSize, self.cellSize, self.boxesPerCell, 4])

        # get real size form 0-1 predicted size
        predictBoxes = predictBoxes * [self.imageSize / self.cellSize,
                                       self.imageSize / self.cellSize,
                                       self.imageSize,
                                       self.imageSize]

        #grid cell coord
        baseBoxes = np.zeros([self.cellSize, self.cellSize, 4])

        for y in range(self.cellSize):
            for x in range(self.cellSize):
                baseBoxes[y, x, :] = [self.imageSize / self.cellSize * x,
                         self.imageSize / self.cellSize * y,
                         0, 0]

        #Make the shape of baseBoxes is the same with predictedBoxes.
        baseBoxes = np.tile(np.resize(baseBoxes,
                                      [self.cellSize, self.cellSize, 1, 4]),[1, 1, self.boxesPerCell, 1])

        # predictBoxes is based on cell, baseBoxes is based on grid cell. Add them to get predicts based on the whole image.
        predictBoxes = baseBoxes + predictBoxes

        #iou for each cell 7 * 7 * 1
        iouPredictTruth = self.iou(predictBoxes, label[0:4])

        # filter out the cells that don't have objects
        C = iouPredictTruth * tf.reshape(response,
                                         [self.cellSize, self.cellSize, 1])

        #
        I = iouPredictTruth * tf.reshape(response, [self.cellSize, self.cellSize, 1])

        #get the maximum iou for each cell's boxes
        maxI = tf.reduce_max(I, 2, keepdims=True)

        # the max iou for the cell contains the center point
        I = tf.cast((I >= maxI), tf.float32) * tf.reshape(response, (self.cellSize, self.cellSize, 1))

        #noI: [cell size, cell size, boxes per cell]
        noI = tf.ones_like(I, dtype=tf.float32) - I

        # B confidences
        pC = predict[:, :, self.numClasses:self.numClasses + self.boxesPerCell]

        #real x center, y center
        x = label[0]
        y = label[1]

        sqrtW = tf.sqrt(tf.abs(label[2]))
        sqrtH = tf.sqrt(tf.abs(label[3]))

        # real predicted x center and y center
        pX = predictBoxes[:, :, :, 0]
        pY = predictBoxes[:, :, :, 1]

        #square root of predicted boxes' width and height
        pSqrtW = tf.sqrt(tf.minimum(self.imageSize * 1.0, tf.maximum(0.0, predictBoxes[:, :, :, 2])))
        pSqrtH = tf.sqrt(tf.minimum(self.imageSize * 1.0, tf.maximum(0.0, predictBoxes[:, :, :, 3])))

        # one hot encoding
        P = tf.one_hot(tf.cast(label[4], tf.int32), self.numClasses, dtype=tf.float32)

        #predict classes
        pP = predict[:, :, 0:self.numClasses]

        #classLoss: only cells containing objects
        classLoss = tf.nn.l2_loss(tf.reshape(objects, (self.cellSize, self.cellSize, 1)) * (pP -P)) * self.classScale

        #objectLoss: object center location loss
        objectLoss = tf.nn.l2_loss(I * (pC - C)) * self.objectScale

        noObjectLoss = tf.nn.l2_loss(noI * (pC)) * self.noobjectScale

        coordLoss = (tf.nn.l2_loss(I * (pX -x)/(self.imageSize/self.cellSize)) +
                     tf.nn.l2_loss(I * (pY -y)/(self.imageSize/self.cellSize)) +
                     tf.nn.l2_loss(I * (pSqrtW - sqrtW))/self.imageSize +
                     tf.nn.l2_loss(I * (pSqrtH - sqrtH))/self.imageSize) + self.coordScale
        nilboy = I

        return num + 1, objectNum, [loss[0] + classLoss, loss[1] + objectLoss, loss[2] + noObjectLoss, loss[3] + coordLoss],predict, labels, nilboy

    def loss(self, predicts, labels, objectsNum):
        '''
        Add loss to all the trainable variables.

        Args:
            predicts: 4-D tensor [batch size, cell size, cell size, 5 * boxes per cell]
            labels: labels: 3-D tensor [batch size, max objects, 5]
            objectNum: 1-D tensor [batch size]
        '''
        classLoss = tf.constant(0, tf.float32)
        objectLoss = tf.constant(0, tf.float32)
        noObjectLoss = tf.constant(0, tf.float32)
        coordLoss = tf.constant(0, tf.float32)

        loss = [0, 0, 0, 0]

        for i in range(self.batchSize):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            objectNum = objectsNum[i]
            nilboy = tf.ones([7, 7, 2])
            tupleResults = tf.while_loop(self.cond1, self.body1,
                                         [tf.constant(0), objectNum, [classLoss, objectLoss, noObjectLoss, coordLoss], predict, label, nilboy])
            for j in range(4):
                loss[j] = loss[j] + tupleResults[2][j]
            nilboy = tupleResults[5]

            tf.add_to_collection('losses', (loss[0] + loss[1] +loss[2] +loss[3])/self.batchSize)

            tf.summary.scalar('class_loss', loss[0]/self.batchSize)
            tf.summary.scalar('object_loss', loss[1]/self.batchSize)
            tf.summary.scalar('noobject_loss', loss[2]/self.batchSize)
            tf.summary.scalar('coord_loss', loss[2]/self.batchSize)
            tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/self.batchSize)

        return tf.add_n(tf.get_collection('losses'), name = 'total_loss'), nilboy
