import os
import sys
import time
import glob
from model import *
from config import *

from keras.models import load_model,save_model
from keras.layers import Input
from keras import optimizers

import dataset
import config

import misc


def load_G_weights(G, path, by_name = True):
    G_path = os.path.join(path,'Generator.h5')
    G.load_weights(G_path, by_name = by_name)
    return G

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown_linear(epoch, num_epochs, rampdown_length):
    if epoch >= num_epochs - rampdown_length:
        return float(num_epochs - epoch) / rampdown_length
    else:
        return 1.0

def create_result_subdir(result_dir, run_desc):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, run_desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print ("Saving results to", result_subdir)
    return result_subdir

def random_latents(num_latents, G_input_shape):
    return np.random.randn(num_latents, *G_input_shape[1:]).astype(np.float32)

def random_labels(num_labels, training_set):
    return training_set.labels[np.random.randint(training_set.labels.shape[0], size=num_labels)]


def predict_gan():
    separate_funcs          = False
    drange_net              = [-1,1]
    drange_viz              = [-1,1]
    image_grid_size         = None
    image_grid_type         = 'default'
    resume_network          = 'pre-trained_weight'
    
    np.random.seed(config.random_seed)

    if resume_network:
        print("Resuming weight from:"+resume_network)
        G = Generator(num_channels=3, resolution=128, label_size=0, **config.G)
        G = load_G_weights(G,resume_network,True)

    print(G.summary())

    # Misc init.

    if image_grid_type == 'default':
        if image_grid_size is None:
            w, h = G.output_shape[1], G.output_shape[2]
            print("w:%d,h:%d"%(w,h))
            image_grid_size = np.clip(int(1920 // w), 3, 16).astype('int'), np.clip(1080 / h, 2, 16).astype('int')
        
        print("image_grid_size:",image_grid_size)
    else:
        raise ValueError('Invalid image_grid_type', image_grid_type)

    result_subdir = misc.create_result_subdir('pre-trained_result', config.run_desc)

    for i in range(1,6):
        snapshot_fake_latents = random_latents(np.prod(image_grid_size), G.input_shape)
        snapshot_fake_images = G.predict_on_batch(snapshot_fake_latents)
        misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'pre-trained_%03d.png'%i), drange=drange_viz, grid_size=image_grid_size)

if __name__ == '__main__':
    predict_gan()
