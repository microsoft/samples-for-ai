#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:44:36 2018

@author: zsc
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

class Net(object):
    '''
    Basic yolo component.
    '''

    def __init__(self, commonParams, netParams):
        '''
        Init the object.

        Args:
            commonParams: a dict of pretrained parameters
            netParams: a dict of trainable parameters
        '''
        self.pretrainedCollection = []
        self.trainableCollection = []

    def _variableInit(self, name, shape, initializer, pretrain=True, train=True):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        if pretrain:
            self.pretrainedCollection.append(var)
        if train:
            self.trainableCollection.append(var)
        return var

    def _variableInitDecay(self, name, shape, stddev, wd, pretrain = True, train = True):
        '''
        Initialize weights and biases.

        Args:
            name: variable nama
            shape: variable shape
            stddev: standard devision of your variables
            wd: L2 loss lambda

        Returns:
            a tensor of variables
        '''

        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
        var = self._variableInit(name,
                                 shape,
                                 initializer=initializer,
                                 pretrain=pretrain,
                                 train=train)
        if wd is not None:
            weightDecay = tf.multiply(tf.nn.l2_loss(var), wd, name='weightLoss')
            tf.add_to_collection('losses', weightDecay)
        return var

    def conv2d(self, scope, input, kernel_size, stride = 1, pretrain = True, train = True):
        '''
        Convolutional layer.

        Args:
            scope: tensorflow scope name
            input: a 4-D tensor [batch size, height, width, channels]
            kernel_size: [height, width, #input channel, #output channel]
            stride: a stride value, int
            pretrain: True or False
            train: True or False

        Return:
            a 4-D tensor by [batch size, height, width, #outpuut channel]
        '''
        with tf.variable_scope(scope) as scope:
            # initialize kernel for a specified layer
            kernel = self._variableInitDecay('weights',
                                        kernel_size,
                                        stddev=5e-2,
                                        wd = self.wd,
                                        pretrain=pretrain,
                                        train = train )

            conv = tf.nn.conv2d(input,
                                kernel,
                                strides=[1, stride, stride, 1],
                                padding='SAME')

            biases = self._variableInit('biases',
                                        kernel_size[3:],
                                        tf.constant_initializer(0.0),
                                        pretrain=pretrain,
                                        train=train)

            convout = self.leakyRelu(tf.nn.bias_add(conv, biases))
        return convout

    def maxPool(self, input, kernel_size, stride):
        '''
        Max pooling layer.

        Args:
            input: a 4-D tensor
            kernel_size: [height, width]
            stride: a int32 number

        Return:
            output: 4-D tensor
        '''
        return tf.nn.max_pool(input,
                              ksize=[1, kernel_size[0], kernel_size[1], 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')

    def fullyConnectLayer(self, scope, input, inDim, outDim, leaky = True, pretrain = True, train = True):
        '''
        Fully connected layer.

        Args:
            scope: variable scope name
            input: input tensor
            inDim: int32
            outDim: int 32

        Return:
            output: a 2-D tensor [batch size, outDim]
        '''
        with tf.variable_scope(scope) as scope:
            reshape = tf.reshape(input,
                                 [tf.shape(input)[0],-1])
            weights = self._variableInitDecay('weights',
                                         [inDim, outDim],
                                         stddev= 0.04,
                                         wd = self.wd,
                                         pretrain=pretrain,
                                         train=train)

            biases = self._variableInit('biases',
                                        [outDim],
                                        tf.constant_initializer(0.0),
                                        pretrain,
                                        train)

            fcnOut = tf.matmul(reshape, weights) + biases

        return fcnOut

    def leakyRelu(self, x, alpha = 0.1, dtype = tf.float32):
        '''
        Leaky relu.
        If x > 0, return x; else, 0.1x.

        Args:
            x: a tensor
            alpha: the specified slope

        Return:
            y: a tensor

        '''
        x = tf.cast(x, dtype = dtype)
        boolMask = (x > 0)
        mask = tf.cast(boolMask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x
