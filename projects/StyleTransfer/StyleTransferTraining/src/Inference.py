import sys
sys.path.insert(0, '.')
import vgg, stylenet
import numpy as np
import argparse
import tensorflow as tf
import os

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('--input', '-i', type=str, help='content image')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--ckpt', '-c', default='c1', type=str, help='checkpoint to be loaded')
parser.add_argument('--out', '-o', default='stylized_image.jpg', type=str, help='stylized image\'s name')

args = parser.parse_args()

outfile_path = args.out
content_image_path = args.input
ckpt = args.ckpt
gpu = args.gpu

original_image = vgg.read_img(content_image_path).astype(np.float32) / 255.0
shaped_input = original_image.reshape((1,) + original_image.shape)


if gpu > -1:
    device = '/gpu:{}'.format(gpu)
else:
    device = '/cpu:0'

with tf.device(device):
    inputs = tf.placeholder(tf.float32, shaped_input.shape, name='input')
    net = stylenet.net(inputs)
    saver = tf.train.Saver(restore_sequentially=True)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:                
        input_checkpoint = tf.train.get_checkpoint_state(ckpt)
        saver.restore(sess, input_checkpoint.model_checkpoint_path)           
        out = sess.run(net, feed_dict={inputs: shaped_input})
    
vgg.save_img(outfile_path, out.reshape(out.shape[1:]))

