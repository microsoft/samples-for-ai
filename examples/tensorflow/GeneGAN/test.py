# -*- coding:utf-8 -*-
# Created Time: Tue 02 May 2017 09:42:27 PM CST
# $Author: Taihong Xiao <xiaotaihong@126.com>

import tensorflow as tf 
import numpy as np 
from model import Model 
from dataset import Dataset 
import os 
import cv2
from scipy import misc 
import argparse



def swap_attribute(src_img, att_img, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out1: src_img with attributes
        out2: att_img without attributes
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu 
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        out2, out1 = sess.run([model.Ae, model.Bx], feed_dict={model.Ax: att_img, model.Be: src_img})
        misc.imsave('out1.jpg', out1[0])
        misc.imsave('out2.jpg', out2[0])


def interpolation(src_img, att_img, inter_num, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        inter_num: number of interpolation points
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: [src_img, inter1, inter2, ..., inter_{inter_num}]
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        out = src_img[0]
        for i in range(1, inter_num + 1):
            lambda_i = i / float(inter_num)
            model.out_i = model.joiner('G_joiner', model.B, model.x * lambda_i) 
            out_i = sess.run(model.out_i, feed_dict={model.Ax: att_img, model.Be: src_img})
            out = np.concatenate((out, out_i[0]), axis=1)
        # print(out.shape)
        misc.imsave('interpolation.jpg', out)

def interpolation2(src_img, att_img, inter_num, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        inter_num: number of interpolation points
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: [src_img, inter1, inter2, ..., inter_{inter_num}]
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        B, src_feat = sess.run([model.B, model.e], feed_dict={model.Be: src_img})
        att_feat = sess.run(model.x, feed_dict={model.Ax: att_img})

        out = src_img[0]
        for i in range(1, inter_num + 1):
            lambda_i = i / float(inter_num)
            out_i = sess.run(model.joiner('G_joiner', B, src_feat + (att_feat - src_feat) * lambda_i) )
            out = np.concatenate((out, out_i[0]), axis=1)
        # print(out.shape)
        misc.imsave('interpolation2.jpg', out)

def interpolation_matrix(src_img, att_imgs, size, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute [1, h, w, c]
        att_imgs: four attribute images that has certain attribute [4, h, w, c]
        size: the size of output matrix 
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out1: image matrix 
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        m, n = size
        h, w = model.height, model.width
        
        rows = [[1 - i/float(m-1), i/float(m-1)] for i in range(m)]
        cols = [[1 - i/float(n-1), i/float(n-1)] for i in range(n)]
        four_tuple = []
        for row in rows:
            for col in cols:
                four_tuple.append([row[0]*col[0], row[0]*col[1], row[1]*col[0], row[1]*col[1]])

        attributes = [sess.run(model.x, feed_dict={model.Ax: att_imgs[i:i+1]}) for i in range(4)]
        B = sess.run(model.B, feed_dict={model.Be: src_img})

        cnt = 0 
        out = np.zeros((0, w * n, model.channel))
        for i in range(m):
            out_row = np.zeros((h, 0, model.channel))
            for j in range(n):
                four = four_tuple[cnt]
                attribute = sum([four[i] * attributes[i] for i in range(4)])
                # print(attribute.shape)
                img = sess.run(model.joiner('G_joiner', B, attribute))[0]
                out_row = np.concatenate((out_row, img), axis=1)
                cnt += 1
            out = np.concatenate((out, out_row), axis=0)

        first_col = np.concatenate((att_imgs[0], 255*np.ones(((m-2)*h, w, 3)), att_imgs[2]), axis=0)

        last_col  = np.concatenate((att_imgs[1], 255*np.ones(((m-2)*h, w, 3)), att_imgs[3]), axis=0)

        out_canvas = np.concatenate((first_col, out, last_col), axis=1)
        misc.imsave('four_matrix.jpg', out_canvas)



def main():
    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--mode', 
        default='swap',
        type=str,
        choices=['swap', 'interpolation', 'matrix'],
        help='Specify mode.'
    )
    parser.add_argument(
        '-i', '--input', 
        type=str,
        help='Specify source image name.'
    )
    parser.add_argument(
        '-t', '--target', 
        metavar='target image with attributes',
        type=str,
        help='Specify target image name.'
    )
    parser.add_argument(
        '--targets', 
        nargs=4,
        type=str,
        help='Specify target image name.'
    )
    parser.add_argument(
        '--model_dir', 
        default='train_log/model/',
        type=str,
        help='Specify model_dir. \ndefault: %(default)s.'
    )
    parser.add_argument(
        '-n', '--num', 
        default='2',
        type=int,
        help='Specify number of interpolations.'
    )
    parser.add_argument(
        '-s', '--size', 
        nargs=2,
        default=[3,3],
        type=int,
        help='Specify number of interpolations.'
    )
    parser.add_argument(
        '-g', '--gpu', 
        default='0',
        type=str,
        help='Specify GPU id. \ndefault: %(default)s. \nUse comma to seperate several ids, for example: 0,1'
    )
    args = parser.parse_args()

    GeneGAN = Model(is_train=False, nhwc=[1,64,64,3])
    if args.mode == 'swap':
        src_img = np.expand_dims(misc.imresize(misc.imread(args.input), (GeneGAN.height, GeneGAN.width)), axis=0)
        att_img = np.expand_dims(misc.imresize(misc.imread(args.target), (GeneGAN.height, GeneGAN.width)), axis=0)
        swap_attribute(src_img, att_img, args.model_dir, GeneGAN, args.gpu)
    elif args.mode == 'interpolation':
        src_img = np.expand_dims(misc.imresize(misc.imread(args.input), (GeneGAN.height, GeneGAN.width)), axis=0)
        att_img = np.expand_dims(misc.imresize(misc.imread(args.target), (GeneGAN.height, GeneGAN.width)), axis=0)
        interpolation(src_img, att_img, args.num,  args.model_dir, GeneGAN, args.gpu)   
    elif args.mode == 'matrix':
        src_img = np.expand_dims(misc.imresize(misc.imread(args.input), (GeneGAN.height, GeneGAN.width)), axis=0)
        att_imgs = np.array([misc.imresize(misc.imread(img), (GeneGAN.height, GeneGAN.width)) for img in args.targets])
        interpolation_matrix(src_img, att_imgs, args.size, args.model_dir, GeneGAN, args.gpu) 
    else:
        raise NotImplementationError()

if __name__ == "__main__":
    main()
