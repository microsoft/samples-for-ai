# -*- coding:utf-8 -*-
# Created Time: Thu 13 Apr 2017 04:07:50 PM CST
# $Author: Taihong Xiao <xiaotaihong@126.com>

import numpy as np 
import tensorflow as tf 
import glob, os, time
from scipy import misc


class Config:
    @property 
    def base_dir(self):
        return os.path.abspath(os.curdir)
         
    @property
    def data_dir(self):
        data_dir = os.path.join(self.base_dir, './datasets/celebA/')
        if not os.path.exists(data_dir):
            raise ValueError('Please specify a data dir.')
        return data_dir
    
    @property
    def exp_dir(self):
        exp_dir = os.path.join(self.base_dir, 'train_log')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir
    
    @property
    def model_dir(self):
        model_dir = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def sample_img_dir(self):
        sample_img_dir = os.path.join(self.exp_dir, 'sample_img')
        if not os.path.exists(sample_img_dir):
            os.makedirs(sample_img_dir)
        return sample_img_dir

    def g_lr(self, init_lr=0.00005, decay_rate=1, decay_step=10000, epoch=0):
        return init_lr * decay_rate ** (epoch / np.float(decay_step))

    def d_lr(self, init_lr=0.00005, decay_rate=1, decay_step=10000, epoch=0):
        return init_lr * decay_rate ** (epoch / np.float(decay_step))

    nhwc = [64,64,64,3]

    num_threads = 10

    capacity = 64000

    shuffle = True

    max_iter = 100000

    weight_decay = 5e-5

    second_ratio = 0.25


config = Config()


class Dataset(object):

    def __init__(self, feature, data_dir=config.data_dir, nhwc=config.nhwc, num_threads=config.num_threads, capacity=config.capacity, shuffle=config.shuffle):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.feature = feature

        self.batch_size, self.height, self.width, self.channel = nhwc
        self.num_threads = num_threads
        self.capacity = capacity
        self.shuffle = shuffle

        # with open(os.path.join(self.data_dir, 'list_landmarks_celeba.txt'), 'r') as f:
        #     self.landmark = [list(map(int, x.split()[1:11])) for x in f.read().strip().split('\n')[2:]]

        with open(os.path.join(self.data_dir, 'list_attr_celeba.txt'), 'r') as f:
            lines = f.read().strip().split('\n')
            col_id = lines[1].split().index(self.feature) + 1
            self.attribute = list(map(int, [x.split()[col_id] for x in lines[2:]]))

        self.idxs1 = list(filter(lambda x: self.attribute[x] ==  1, range(len(self.attribute))))
        self.idxs2 = list(filter(lambda x: self.attribute[x] == -1, range(len(self.attribute))))

    @property
    def filenames1(self):
        filenames = [os.path.join(self.data_dir, 'align_5p/{:06d}.jpg'.format(idx+1)) for idx in self.idxs1]
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)
        return filenames

    @property
    def filenames2(self):
        filenames = [os.path.join(self.data_dir, 'align_5p/{:06d}.jpg'.format(idx+1)) for idx in self.idxs2]
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)
        return filenames

    def read_images(self, input_queue):
        reader = tf.WholeFileReader()
        filename, content = reader.read(input_queue)
        image = tf.image.decode_jpeg(content, channels=self.channel)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, size=[self.height,self.width])
        return image 
        
    def input(self):
        input_queue1 = tf.train.string_input_producer(self.filenames1)
        image1 = self.read_images(input_queue1)

        input_queue2 = tf.train.string_input_producer(self.filenames2)
        image2 = self.read_images(input_queue2)

        if self.shuffle:
            batch1, batch2 = tf.train.shuffle_batch([image1,image2], 
                                    batch_size=self.batch_size, 
                                    capacity=self.capacity,
                                    num_threads=self.num_threads,
                                    min_after_dequeue=256
                                    )
        else:
            batch1, batch2 = tf.train.batch([image1,image2],
                                    batch_size=self.batch_size, 
                                    capacity=self.capacity,
                                    num_threads=self.num_threads
                                    )
        return batch1, batch2


if __name__ == '__main__':  
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    config = Config()
    celebA = Dataset('Bangs')

    batch1, batch2 = celebA.input()


    X1 = tf.placeholder(tf.float32, config.nhwc)
    X2 = tf.placeholder(tf.float32, config.nhwc)
    Y = tf.reduce_mean(X1) + tf.reduce_mean(X2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    t1 = time.time()
    for i in range(100):
        print(i, sess.run(Y, feed_dict={X1: sess.run(batch1), X2: sess.run(batch2)}))
        # sess.run([batch1, batch2])
    t2 = time.time()
    print(t2-t1, (t2-t1)/10)


    coord.request_stop()
    coord.join(threads)
    



