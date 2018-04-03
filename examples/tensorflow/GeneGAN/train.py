# -*- coding:utf-8 -*-
# Created Time: Thu 13 Apr 2017 04:07:50 PM CST
# $Author: Taihong Xiao <xiaotaihong@126.com>

import tensorflow as tf 
import os
from model import Model
from dataset import config, Dataset
import numpy as np
from scipy import misc
import argparse


def run(config, dataset, model, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    batch1, batch2 = dataset.input()

    saver = tf.train.Saver()

    # image summary
    Ax_op  = tf.summary.image('Ax', model.Ax, max_outputs=30)
    Be_op  = tf.summary.image('Be', model.Be, max_outputs=30)
    Ax2_op = tf.summary.image('Ax2', model.Ax2, max_outputs=30)
    Be2_op = tf.summary.image('Be2', model.Be2, max_outputs=30)
    Bx_op  = tf.summary.image('Bx', model.Bx, max_outputs=30)
    Ae_op  = tf.summary.image('Ae', model.Ae, max_outputs=30)

    # G loss summary
    for key in model.G_loss.keys():
        tf.summary.scalar(key, model.G_loss[key])

    loss_G_nodecay_op = tf.summary.scalar('loss_G_nodecay', model.loss_G_nodecay)
    loss_G_decay_op = tf.summary.scalar('loss_G_decay', model.loss_G_decay)
    loss_G_op = tf.summary.scalar('loss_G', model.loss_G)

    # D loss summary
    for key in model.D_loss.keys():
        tf.summary.scalar(key, model.D_loss[key])

    loss_D_op = tf.summary.scalar('loss_D', model.loss_D)

    # learning rate summary
    g_lr_op = tf.summary.scalar('g_learning_rate', model.g_lr)
    d_lr_op = tf.summary.scalar('d_learning_rate', model.d_lr)

    merged_op = tf.summary.merge_all()

    # start training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    
    writer = tf.summary.FileWriter(config.log_dir, sess.graph)
    writer.add_graph(sess.graph)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(config.max_iter):        
        d_num = 100 if i % 500 == 0 else 1

        # update D with clipping 
        for j in range(d_num):
            _, loss_D_sum, _ = sess.run([model.d_opt, model.loss_D, model.clip_d], 
                    feed_dict ={model.Ax: sess.run(batch1), 
                                model.Be: sess.run(batch2), 
                                model.g_lr: config.g_lr(epoch=i), 
                                model.d_lr: config.d_lr(epoch=i)
                                })

        # update G 
        _, loss_G_sum = sess.run([model.g_opt, model.loss_G],
                    feed_dict ={model.Ax: sess.run(batch1), 
                                model.Be: sess.run(batch2), 
                                model.g_lr: config.g_lr(epoch=i), 
                                model.d_lr: config.d_lr(epoch=i)
                                })

        print('iter: {:06d},   g_loss: {}    d_loss: {}'.format(i, loss_D_sum, loss_G_sum))

        if i % 20 == 0:
            merged_summary = sess.run(merged_op, 
                    feed_dict ={model.Ax: sess.run(batch1), 
                                model.Be: sess.run(batch2),  
                                model.g_lr: config.g_lr(epoch=i), 
                                model.d_lr: config.d_lr(epoch=i)
                                })

            writer.add_summary(merged_summary, i)

        if i % 500 == 0:
            saver.save(sess, os.path.join(config.model_dir, 'model_{:06d}.ckpt'.format(i)))

            img_Ax, img_Be, img_Ae, img_Bx, img_Ax2, img_Be2 = sess.run([model.Ax, model.Be, model.Ae, model.Bx, model.Ax2, model.Be2],
                                                        feed_dict={model.Ax: sess.run(batch1), model.Be: sess.run(batch2)})          

            for j in range(5):                                                                                                                                              
                img = np.concatenate((img_Ax[j], img_Be[j], img_Ae[j], img_Bx[j], img_Ax2[j], img_Be2[j]), axis=1)                                      
                misc.imsave(os.path.join(config.sample_img_dir, 'iter_{:06d}_{}.jpg'.format(i,j)), img)
        
    writer.close()
    saver.save(sess, os.path.join(config.model_dir, 'model.ckpt'))

    coord.request_stop()
    coord.join(threads)

def main():
    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--attribute', 
        default='Smiling',
        type=str,
        help='Specify attribute name for training. \ndefault: %(default)s. \nAll attributes can be found in list_attr_celeba.txt'
    )
    parser.add_argument(
        '-g', '--gpu', 
        default='0',
        type=str,
        help='Specify GPU id. \ndefault: %(default)s. \nUse comma to seperate several ids, for example: 0,1'
    )
    args = parser.parse_args()

    celebA = Dataset(args.attribute)
    GeneGAN = Model(is_train=True)
    run(config, celebA, GeneGAN, gpu=args.gpu)


if __name__ == "__main__":
    main()
