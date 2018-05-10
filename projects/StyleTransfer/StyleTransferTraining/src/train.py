from __future__ import print_function
import sys, pdb
sys.path.insert(0, '.')
import vgg, time
import tensorflow as tf, numpy as np, os
import stylenet
from argparse import ArgumentParser
from vgg import read_img, list_files

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def build_parser():
    parser = ArgumentParser(description='Real-time style transfer')
    parser.add_argument("--model_dir", '-model', help="Final model directory.")
    parser.add_argument("--log_dir", '-log', help="TensorBoard log directory.")
    parser.add_argument('--gpu', '-g', type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', default='../data', type=str,
                        help='Top-level data directory path (according to the paper, use MSCOCO 80k images)')
    parser.add_argument('--style_image', '-s', default='starry_night.jpg' ,type=str,
                        help='style image')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='batch size (default value is 1)')
    parser.add_argument('--ckpt', '-c', type=str,
                        help='the global step of checkpoint file desired to restore.')
    parser.add_argument('--lambda_tv', '-l_tv', default=2e2, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--lambda_feat', '-l_feat', default=7.5e0, type=float)
    parser.add_argument('--lambda_style', '-l_style', default=15, type=float)
    parser.add_argument('--epoch', '-e', default=2, type=int)
    parser.add_argument('--lr', '-l', default=1e-3, type=float)

    return parser
    

def main():
    parser = build_parser()
    options, unknown = parser.parse_known_args()
    env = os.environ.copy()

    print("options: ", options)
    vgg_path = options.dataset + '/vgg/imagenet-vgg-verydeep-19.mat'
    style_image = options.dataset + '/style_images/' + options.style_image
    training_path = options.dataset + '/train'


    if options.model_dir == None:
        options.model_dir = options.dataset + '/model'

    model_dir = options.model_dir
    if os.path.isfile(model_dir):
        print("model directory: {0} should not be a file.".format(model_dir))
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_name = options.style_image.replace('.jpg', '.ckpt')


    if options.log_dir == None:
        options.log_dir = options.dataset + '/log'

    log_dir = options.log_dir
    if os.path.isfile(log_dir):
        print("log directory: {0} should not be a file.".format(log_dir))
        return

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("style image path: ", style_image)
    print("vgg path: ", vgg_path)
    print("training image path: ", training_path)
    print("model path: ", model_name)
    print("log path: ", log_dir)

    if not os.path.isfile(vgg_path):
        print('Pre-trained VGG model: {0} does not exist.'.format(vgg_path))
        print('Plese download it from http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat')
        return;

    if not os.path.isdir(options.training_path):
        print('Training image directory: {0} does not exist.'.format(training_path))
        print('Plese download COCO 2014 train images from http://images.cocodataset.org/zips/train2014.zip')
        return;

    device = '/gpu:0'
    if options.gpu == None:
        available_gpus = get_available_gpus()
        if len(available_gpus) > 0:
            device = '/gpu:0'
    else:
        if options.gpu > -1:
            device = '/gpu:{}'.format(options.gpu)

    batchsize = options.batchsize

    # content targets
    content_targets = [os.path.join(training_path, fn) for fn in list_files(training_path)]
    if len(content_targets) % batchsize != 0:
        content_targets = content_targets[:-(len(content_targets) % batchsize)]

    print('total training data size: ', len(content_targets))
    batch_shape = (batchsize,224,224,3)

    # style target
    style_target = read_img(style_image)
    style_shape = (1,) + style_target.shape

    with tf.device(device), tf.Session() as sess:

        # style target feature
        # compute gram maxtrix of style target
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        vggstyletarget = vgg.net(vgg_path, vgg.preprocess(style_image))
        style_vgg = vgg.get_style_vgg(vggstyletarget, style_image, np.array([style_target]))        

        # content target feature 
        content_vgg = {}
        inputs = tf.placeholder(tf.float32, shape=batch_shape, name="inputs")
        content_net = vgg.net(vgg_path, vgg.preprocess(inputs))
        content_vgg['relu4_2'] = content_net['relu4_2']

        # feature after transformation 
        outputs = stylenet.net(inputs/255.0)        
        vggoutputs = vgg.net(vgg_path, vgg.preprocess(outputs))

        # compute feature loss
        loss_f = options.lambda_feat * vgg.total_content_loss(vggoutputs, content_vgg, batchsize)

        # compute style loss        
        loss_s = options.lambda_style * vgg.total_style_loss(vggoutputs, style_vgg, batchsize)
        
        # total variation denoising
        loss_tv = options.lambda_tv * vgg.total_variation_regularization(outputs, batchsize, batch_shape)
        
        # total loss
        loss = loss_f + loss_s + loss_tv

        
    with tf.Session() as sess:    
                
        if not os.path.exists(options.ckpt):
            os.makedirs(options.ckpt)

        save_path = model_dir + '/' + model_name

        # op to write logs to Tensorboard

        #training
        train_step = tf.train.AdamOptimizer(options.lr).minimize(loss)
        sess.run(tf.global_variables_initializer())        
    
        total_step = 0
        for epoch in range(options.epoch):
            print('epoch: ', epoch)
            step = 0
            while step * batchsize < len(content_targets):
                time_start = time.time()
                
                batch = np.zeros(batch_shape, dtype=np.float32)
                for i, img in enumerate(content_targets[step * batchsize : (step + 1) * batchsize]):
                   batch[i] = read_img(img).astype(np.float32) # (224,224,3)

                step += 1
                total_step += 1
            
                loss_, summary= sess.run([loss, train_step,], feed_dict= {inputs:batch})
                
             
                time_elapse = time.time() - time_start
                
                should_save = total_step % 2000 == 0                
               
                if total_step % 1 == 0:
                  
                    print('[step {}] elapse time: {} loss: {}'.format(total_step, time_elapse, loss_))

                if should_save:                                        
                    print('Saving checkpoint')
                    saver = tf.train.Saver()
                    res = saver.save(sess, save_path)
        
        print('Saving final result to ' + save_path)
        saver = tf.train.Saver()
        res = saver.save(sess, save_path)


if __name__ == '__main__':
    main()
