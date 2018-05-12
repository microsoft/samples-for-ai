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

def load_GD(path, compile = False):
    G_path = os.path.join(path,'Generator.h5')
    D_path = os.path.join(path,'Discriminator.h5')
    G = load_model(G_path, compile = compile)
    D = load_model(D_path, compile = compile)
    return G,D

def save_GD(G,D,path,overwrite = False):

        os.makedirs(path);
        G_path = os.path.join(path,'Generator.h5')
        D_path = os.path.join(path,'Discriminator.h5')
        save_model(G,G_path,overwrite = overwrite)
        save_model(D,D_path,overwrite = overwrite)
        print("Save model to %s"%path)


def load_GD_weights(G,D,path, by_name = True):
    G_path = os.path.join(path,'Generator.h5')
    D_path = os.path.join(path,'Discriminator.h5')
    G.load_weights(G_path, by_name = by_name)
    D.load_weights(D_path, by_name = by_name)
    return G,D

def save_GD_weights(G,D,path):
    try:
        os.makedirs(path);
        G_path = os.path.join(path,'Generator.h5')
        D_path = os.path.join(path,'Discriminator.h5')
        G.save_weights(G_path)
        D.save_weights(D_path)
        print("Save weights to %s:"%path)
    except:
        print("Save model snapshot failed!")


def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def format_time(seconds):
    s = int(np.round(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s / 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s / (60*60), (s / 60) % 60, s % 60)
    else:              return '%dd %dh %02dm'      % (s / (24*60*60), (s / (60*60)) % 24, (s / 60) % 60)

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

def wasserstein_loss( y_true, y_pred):
        return K.mean(y_true * y_pred)

def multiple_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def mean_loss(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset(dataset_spec=None, verbose=True, **spec_overrides):
    if verbose: print('Loading dataset...')
    if dataset_spec is None: dataset_spec = config.dataset
    dataset_spec = dict(dataset_spec) # take a copy of the dict before modifying it
    dataset_spec.update(spec_overrides)
    dataset_spec['h5_path'] = os.path.join(config.data_dir, dataset_spec['h5_path'])
    if 'label_path' in dataset_spec: dataset_spec['label_path'] = os.path.join(config.data_dir, dataset_spec['label_path'])
    training_set = dataset.Dataset(**dataset_spec)
    if verbose: print('Dataset shape =', np.int32(training_set.shape).tolist())
    drange_orig = training_set.get_dynamic_range()
    if verbose: print('Dynamic range =', drange_orig)
    return training_set, drange_orig



speed_factor = 20

def train_gan(
    separate_funcs          = False,
    D_training_repeats      = 1,
    G_learning_rate_max     = 0.0010,
    D_learning_rate_max     = 0.0010,
    G_smoothing             = 0.999,
    adam_beta1              = 0.0,
    adam_beta2              = 0.99,
    adam_epsilon            = 1e-8,
    minibatch_default       = 16,
    minibatch_overrides     = {},
    rampup_kimg             = 40/speed_factor,
    rampdown_kimg           = 0,
    lod_initial_resolution  = 4,
    lod_training_kimg       = 400/speed_factor,
    lod_transition_kimg     = 400/speed_factor,
    total_kimg              = 10000/speed_factor,
    dequantize_reals        = False,
    gdrop_beta              = 0.9,
    gdrop_lim               = 0.5,
    gdrop_coef              = 0.2,
    gdrop_exp               = 2.0,
    drange_net              = [-1,1],
    drange_viz              = [-1,1],
    image_grid_size         = None,
    tick_kimg_default       = 50/speed_factor,
    tick_kimg_overrides     = {32:20, 64:10, 128:10, 256:5, 512:2, 1024:1},
    image_snapshot_ticks    = 1,
    network_snapshot_ticks  = 4,
    image_grid_type         = 'default',
    #resume_network          = '000-celeba/network-snapshot-000488',
    resume_network          = None,
    resume_kimg             = 0.0,
    resume_time             = 0.0):

    training_set, drange_orig = load_dataset()


    if resume_network:
        print("Resuming weight from:"+resume_network)
        G = Generator(num_channels=training_set.shape[3], resolution=training_set.shape[1], label_size=training_set.labels.shape[1], **config.G)
        D = Discriminator(num_channels=training_set.shape[3], resolution=training_set.shape[1], label_size=training_set.labels.shape[1], **config.D)
        G,D = load_GD_weights(G,D,os.path.join(config.result_dir,resume_network),True)
    else:
        G = Generator(num_channels=training_set.shape[3], resolution=training_set.shape[1], label_size=training_set.labels.shape[1], **config.G)
        D = Discriminator(num_channels=training_set.shape[3], resolution=training_set.shape[1], label_size=training_set.labels.shape[1], **config.D)
        
    G_train,D_train = PG_GAN(G,D,config.G['latent_size'],0,training_set.shape[1],training_set.shape[3]) 
 
    print(G.summary())
    print(D.summary())


    # Misc init.
    resolution_log2 = int(np.round(np.log2(G.output_shape[2])))
    initial_lod = max(resolution_log2 - int(np.round(np.log2(lod_initial_resolution))), 0)
    cur_lod = 0.0
    min_lod, max_lod = -1.0, -2.0
    fake_score_avg = 0.0

   

    G_opt = optimizers.Adam(lr = 0.0,beta_1=adam_beta1,beta_2=adam_beta2,epsilon = adam_epsilon)
    D_opt = optimizers.Adam(lr = 0.0,beta_1 = adam_beta1,beta_2 = adam_beta2,epsilon = adam_epsilon)
    
    if config.loss['type']=='wass':
        G_loss = wasserstein_loss
        D_loss = wasserstein_loss
    elif config.loss['type']=='iwass':
        G_loss = multiple_loss
        D_loss = [mean_loss,'mse']
        D_loss_weight = [1.0, config.loss['iwass_lambda']]

    G.compile(G_opt,loss=G_loss)
    D.trainable = False
    G_train.compile(G_opt,loss = G_loss)
    D.trainable = True
    D_train.compile(D_opt,loss=D_loss,loss_weights=D_loss_weight)


    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    tick_train_out = []
    train_start_time = tick_start_time - resume_time


    if image_grid_type == 'default':
        if image_grid_size is None:
            w, h = G.output_shape[1], G.output_shape[2]
            print("w:%d,h:%d"%(w,h))
            image_grid_size = np.clip(int(1920 // w), 3, 16).astype('int'), np.clip(1080 / h, 2, 16).astype('int')
        
        print("image_grid_size:",image_grid_size)

        example_real_images, snapshot_fake_labels = training_set.get_random_minibatch_channel_last(np.prod(image_grid_size), labels=True)
        snapshot_fake_latents = random_latents(np.prod(image_grid_size), G.input_shape)
    else:
        raise ValueError('Invalid image_grid_type', image_grid_type)


    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)



    print("example_real_images.shape:",example_real_images.shape)
    misc.save_image_grid(example_real_images, os.path.join(result_subdir, 'reals.png'), drange=drange_orig, grid_size=image_grid_size)


    snapshot_fake_latents = random_latents(np.prod(image_grid_size), G.input_shape)
    snapshot_fake_images = G.predict_on_batch(snapshot_fake_latents)
    misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg / 1000)), drange=drange_viz, grid_size=image_grid_size)
    
    nimg_h = 0
   
    while cur_nimg < total_kimg * 1000:
        
        # Calculate current LOD.
        cur_lod = initial_lod
        if lod_training_kimg or lod_transition_kimg:
            tlod = (cur_nimg / (1000.0/speed_factor)) / (lod_training_kimg + lod_transition_kimg)
            cur_lod -= np.floor(tlod)
            if lod_transition_kimg:
                cur_lod -= max(1.0 + (np.fmod(tlod, 1.0) - 1.0) * (lod_training_kimg + lod_transition_kimg) / lod_transition_kimg, 0.0)
            cur_lod = max(cur_lod, 0.0)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        minibatch_size = minibatch_overrides.get(cur_res, minibatch_default)
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)


        # Update network config.
        lrate_coef = rampup(cur_nimg / 1000.0, rampup_kimg)
        lrate_coef *= rampdown_linear(cur_nimg / 1000.0, total_kimg, rampdown_kimg)

        K.set_value(G.optimizer.lr, np.float32(lrate_coef * G_learning_rate_max))
        K.set_value(G_train.optimizer.lr, np.float32(lrate_coef * G_learning_rate_max))

        K.set_value(D_train.optimizer.lr, np.float32(lrate_coef * D_learning_rate_max))
        if hasattr(G_train, 'cur_lod'): K.set_value(G_train.cur_lod,np.float32(cur_lod))
        if hasattr(D_train, 'cur_lod'): K.set_value(D_train.cur_lod,np.float32(cur_lod))


        new_min_lod, new_max_lod = int(np.floor(cur_lod)), int(np.ceil(cur_lod))
        if min_lod != new_min_lod or max_lod != new_max_lod:
            min_lod, max_lod = new_min_lod, new_max_lod


        # train D
        d_loss = None
        for idx in range(D_training_repeats):
            mb_reals, mb_labels = training_set.get_random_minibatch_channel_last(minibatch_size, lod=cur_lod, shrink_based_on_lod=True, labels=True)
            mb_latents = random_latents(minibatch_size,G.input_shape)
            mb_labels_rnd = random_labels(minibatch_size,training_set)
            if min_lod > 0: # compensate for shrink_based_on_lod
                 mb_reals = np.repeat(mb_reals, 2**min_lod, axis=1)
                 mb_reals = np.repeat(mb_reals, 2**min_lod, axis=2)

            mb_fakes = G.predict_on_batch([mb_latents])

            epsilon = np.random.uniform(0, 1, size=(minibatch_size,1,1,1))
            interpolation = epsilon*mb_reals + (1-epsilon)*mb_fakes
            mb_reals = misc.adjust_dynamic_range(mb_reals, drange_orig, drange_net)
            d_loss, d_diff, d_norm = D_train.train_on_batch([mb_fakes, mb_reals, interpolation], [np.ones((minibatch_size, 1,1,1)),np.ones((minibatch_size, 1))])
            d_score_real = D.predict_on_batch(mb_reals)
            d_score_fake = D.predict_on_batch(mb_fakes)
            print("real score: %d fake score: %d"%(np.mean(d_score_real),np.mean(d_score_fake)))
            cur_nimg += minibatch_size

        #train G

        mb_latents = random_latents(minibatch_size,G.input_shape)
        mb_labels_rnd = random_labels(minibatch_size,training_set)


        g_loss = G_train.train_on_batch([mb_latents], (-1)*np.ones((mb_latents.shape[0],1,1,1)))

        print ("%d [D loss: %f] [G loss: %f]" % (cur_nimg, d_loss,g_loss))



        fake_score_cur = np.clip(np.mean(d_loss), 0.0, 1.0)
        fake_score_avg = fake_score_avg * gdrop_beta + fake_score_cur * (1.0 - gdrop_beta)
        gdrop_strength = gdrop_coef * (max(fake_score_avg - gdrop_lim, 0.0) ** gdrop_exp)
        if hasattr(D, 'gdrop_strength'): K.set_value(D.gdrop_strength,np.float32(gdrop_strength))

        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or cur_nimg >= total_kimg * 1000:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            tick_start_time = cur_time
            tick_train_avg = tuple(np.mean(np.concatenate([np.asarray(v).flatten() for v in vals])) for vals in zip(*tick_train_out))
            tick_train_out = []



            # Visualize generated images.
            if cur_tick % image_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                snapshot_fake_images = G.predict_on_batch(snapshot_fake_latents)
                misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg / 1000)), drange=drange_viz, grid_size=image_grid_size)

            if cur_tick % network_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                save_GD_weights(G,D,os.path.join(result_subdir, 'network-snapshot-%06d' % (cur_nimg / 1000)))


    save_GD(G,D,os.path.join(result_subdir, 'network-final'))
    training_set.close()
    print('Done.')

if __name__ == '__main__':

    np.random.seed(config.random_seed)
    func_params = config.train

    func_name = func_params['func']
    del func_params['func']
    globals()[func_name](**func_params)