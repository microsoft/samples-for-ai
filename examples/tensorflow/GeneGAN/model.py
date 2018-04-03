# -*- coding:utf-8 -*-
# Created Time: Thu 13 Apr 2017 04:07:50 PM CST
# $Author: Taihong Xiao <xiaotaihong@126.com>

from __future__ import print_function
import os
import tensorflow as tf 
import numpy as np 
from dataset import config, Dataset
from six.moves import reduce


class Model(object):
    def __init__(self, is_train=True, nhwc=config.nhwc, max_iter=config.max_iter, weight_decay=config.weight_decay, second_ratio=config.second_ratio):
        super(Model, self).__init__()
        self.is_train = is_train
        self.batch_size, self.height, self.width, self.channel = nhwc
        self.max_iter = max_iter
        self.g_lr = tf.placeholder(tf.float32)
        self.d_lr = tf.placeholder(tf.float32)
        self.weight_decay = weight_decay
        self.second_ratio = second_ratio
        self.reuse = {}
        self.build_model()

    def leakyRelu(self, x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def make_conv(self, name, X, shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W', 
                                shape=shape, 
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.nn.conv2d(X, W, strides=strides, padding='SAME')
                   

    def make_conv_bn(self, name, X, shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W', 
                                shape=shape, 
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.layers.batch_normalization(
                        tf.nn.conv2d(X, W, strides=strides, padding='SAME'),
                        training=self.is_train
                    )
                    
    def make_fc(self, name, X, out_dim):
        in_dim = X.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=[in_dim, out_dim],
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            b = tf.get_variable('b',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                                )
            return tf.add(tf.matmul(X, W), b)

    def make_fc_bn(self, name, X, out_dim):
        in_dim = X.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W', 
                                shape=[in_dim, out_dim],
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            b = tf.get_variable('b',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                                )
            X = tf.add(tf.matmul(X, W), b)
            return tf.layers.batch_normalization(X, training=self.is_train)

    def make_deconv(self, name, X, filter_shape, out_shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=filter_shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.nn.conv2d_transpose(X, W, output_shape=out_shape, strides=strides, padding='SAME')

    def make_deconv_bn(self, name, X, filter_shape, out_shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=filter_shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.layers.batch_normalization(
                        tf.nn.conv2d_transpose(X, W, 
                            output_shape=out_shape, strides=strides, padding='SAME'
                        ), training=self.is_train
                    )

    def discriminator(self, name, image):
        X = image / 255.0
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:   
            X = self.make_conv('conv1', X, shape=[4,4,3,128], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            X = self.make_conv_bn('conv2', X, shape=[4,4,128,256], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            X = self.make_conv_bn('conv3', X, shape=[4,4,256,512], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            X = self.make_conv_bn('conv4', X, shape=[4,4,512,512], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            flat_dim = reduce(lambda x,y: x*y, X.get_shape().as_list()[1:])
            X = tf.reshape(X, [-1, flat_dim])
            X = self.make_fc('fct', X, 1)
            # X = tf.nn.sigmoid(X)
            return X

    def splitter(self, name, image):
        X = image / 255.0
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            X = self.make_conv('conv1', X, shape=[4,4,3,128], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)

            X = self.make_conv_bn('conv2', X, shape=[4,4,128,256], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)

            X = self.make_conv_bn('conv3', X, shape=[4,4,256,512], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)

            num_ch = int(X.get_shape().as_list()[-1] * self.second_ratio)
            return X[:,:,:,:-num_ch], X[:,:,:,-num_ch:]

    def joiner(self, name, A, x):
        X = tf.concat([A, x], axis=-1)
        # X0 = X
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            X = self.make_deconv_bn('deconv1', X, filter_shape=[4,4,512,512], 
                                    out_shape=[self.batch_size, int(self.height/4), int(self.width/4), 512], 
                                    strides=[1,2,2,1])
            X = tf.nn.relu(X)

            X = self.make_deconv_bn('deconv2', X, filter_shape=[4,4,256,512], 
                                    out_shape=[self.batch_size, int(self.height/2), int(self.width/2), 256], 
                                    strides=[1,2,2,1])
            X = tf.nn.relu(X)

            X = self.make_deconv('deconv3', X, filter_shape=[4,4,self.channel,256], 
                                    out_shape=[self.batch_size, self.height, self.width, self.channel], 
                                    strides=[1,2,2,1])
            b = tf.get_variable('b', shape=[1,1,1,self.channel], initializer=tf.zeros_initializer())
            X = X + b

            X = (tf.tanh(X) + 1) * 255.0 / 2
            return X 


    def build_model(self):
        self.Ax = tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channel], name='data1')
        self.Be = tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channel], name='data2')

        self.A, self.x = self.splitter('G_splitter', self.Ax)
        self.B, self.e = self.splitter('G_splitter', self.Be)

        self.Ax2 = self.joiner('G_joiner', self.A, self.x)
        self.Be2 = self.joiner('G_joiner', self.B, tf.zeros_like(self.e))

        # crossover
        self.Bx = self.joiner('G_joiner', self.B, self.x)
        self.Ae = self.joiner('G_joiner', self.A, tf.zeros_like(self.e))

        self.real_Ax = self.discriminator('D_Ax', self.Ax)
        self.fake_Bx = self.discriminator('D_Ax', self.Bx)

        self.real_Be = self.discriminator('D_Be', self.Be)
        self.fake_Ae = self.discriminator('D_Be', self.Ae)


        # variable list
        self.g_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_joiner') \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_splitter')         

        self.d_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_Ax') \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_Be')  


        # G loss
        self.G_loss = {}
        self.G_loss['e'] = tf.reduce_mean(tf.abs(self.e))
        self.G_loss['cycle_Ax'] = tf.reduce_mean(tf.abs(self.Ax - self.Ax2)) / 255.0
        self.G_loss['cycle_Be'] = tf.reduce_mean(tf.abs(self.Be - self.Be2)) / 255.0
        self.G_loss['Bx'] = -tf.reduce_mean(self.fake_Bx)
        self.G_loss['Ae'] = -tf.reduce_mean(self.fake_Ae)
        self.G_loss['parallelogram'] = 0.01 * tf.reduce_mean(tf.abs(self.Ax + self.Be - self.Bx - self.Ae))
        self.loss_G_nodecay = sum(self.G_loss.values())

        self.loss_G_decay = 0.0
        for w in self.g_var_list:
            if w.name.startswith('G') and w.name.endswith('W:0'):
                self.loss_G_decay += 0.5 * self.weight_decay * tf.reduce_mean(tf.square(w))
                # print(w.name)

        self.loss_G = self.loss_G_decay + self.loss_G_nodecay

        # D loss
        self.D_loss = {}
        self.D_loss['Ax_Bx'] = tf.reduce_mean(self.fake_Bx - self.real_Ax)
        self.D_loss['Be_Ae'] = tf.reduce_mean(self.fake_Ae - self.real_Be)
        self.loss_D = sum(self.D_loss.values())


        # G, D optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
            self.g_opt = tf.train.RMSPropOptimizer(self.g_lr, decay=0.8).minimize(self.loss_G, var_list=self.g_var_list)
            self.d_opt = tf.train.RMSPropOptimizer(self.d_lr, decay=0.8).minimize(self.loss_D, var_list=self.d_var_list)

        # clip weights in D
        with tf.name_scope('clip_d'):
            self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_var_list]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    celebA = Dataset('Eyeglasses')
    image_batch = celebA.input()

    GeneGAN = Model()


