#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:05:37 2018

@author: root
"""
import tensorflow as tf
import sys
import time
import numpy as np
import os
from datetime import datetime


class YoloSolver:
    def __init__(self, dataset, net, commonParams, solverParams):
        '''
        Process the paramters
        '''
        self.moment = float(solverParams['moment'])
        self.learningRate = float(solverParams['learning_rate'])
        self.batchSize = int(commonParams['batch_size'])
        self.height = int(commonParams['image_size'])
        self.width = int(commonParams['image_size'])
        self.maxObjects = int(commonParams['max_objects_per_image'])
        self.pretrainPath = str(solverParams['pretrain_model_path'])
        self.trainDir = str(solverParams['train_dir'])
        self.maxIterators = int(solverParams['max_iterators'])

        self.dataset = dataset
        self.net = net
        self.constructGraph()

    def _train(self):
        '''
        Train the model.
        '''
        opt = tf.train.MomentumOptimizer(self.learningRate, self.moment)
        grads = opt.compute_gradients(self.totalLoss)
        applyGradientOp = opt.apply_gradients(grads, self.globalStep)

        return applyGradientOp

    def constructGraph(self):
        '''
        Construct graph
        '''
        self.globalStep = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, (self.batchSize, self.height, self.width, 3))
        self.labels = tf.placeholder(tf.float32, (self.batchSize, self.maxObjects, 5))
        self.objectsNum = tf.placeholder(tf.int32, (self.batchSize))

        self.predicts = self.net.yoloTinyModel(self.images)
        self.totalLoss, self.nilboy = self.net.loss(self.predicts, self.labels, self.objectsNum)
        tf.summary.scalar('loss', self.totalLoss)
        self.trainOp = self._train()

    def solve(self):

        saver1 = tf.train.Saver(self.net.pretrainedCollection, write_version=1)
        saver2 = tf.train.Saver(self.net.trainableCollection, write_version=1)

        init = tf.global_variables_initializer()
        summaryOp = tf.summary.merge_all()
        
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        sess = tf.Session(config = config)
        sess.run(init)

        saver1.restore(sess, self.pretrainPath)
        summaryWriter = tf.summary.FileWriter(self.trainDir, sess.graph)

        for step in range(self.maxIterators):
            startTime = time.time()
            npImages, npLabels, npObjectsNum = self.dataset.batch()

            _, lossValue, nilboy = sess.run([self.trainOp, self.totalLoss, self.nilboy],
                                            feed_dict={self.images: npImages, self.labels: npLabels, self.objectsNum: npObjectsNum})

            duration = time.time() - startTime
            assert not np.isnan(lossValue), 'Model deverged with loss = NaN'
            if step%10 == 0:
                numExamplesPerStep = self.dataset.batchSize
                examplesPerSec = numExamplesPerStep / duration
                secPerBatch = float(duration)

                print('time: '+str(datetime.now())+', step: '+str(step)+', loss: '+str(lossValue)+', examplePerSec: '+str(examplesPerSec)+', secPerBatch: '+str(secPerBatch))
                sys.stdout.flush()

            if step % 100 == 0:
                summaryStr = sess.run(summaryOp,
                                      feed_dict={self.images: npImages, self.labels:npLabels, self.objectsNum: npObjectsNum})

                summaryWriter.add_summary(summaryStr, step)

            if step % 5000 == 0:
                saver2.save(sess, self.trainDir + 'model.ckpt', global_step=step)
        sess.close()
