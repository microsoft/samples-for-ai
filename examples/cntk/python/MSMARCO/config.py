data_config = {
    'word_size' : 50,
    'word_count_threshold' : 10,
    'char_count_threshold' : 50,
    'pickle_file' : 'vocabs.pkl',
    'word_embed_file': 'glove.840B.300d.txt',
    'char_embed_file': 'glove.840B.300d-char.txt'
}

model_config = {
    'hidden_dim'     	: 90,
    'char_convs'     	: 90,
    'char_emb_dim'   	: 300,
    'word_emb_dim'      : 300,
    'dropout'        	: 0.5,
    'highway_layers' 	: 2,
    'two_step'          : True,
    'use_cudnn'         : True,
    'use_layerbn'       : False
}

def create_learner(paras, lr):
    import cntk as C
    return C.adadelta(paras, lr, 0.95, 1e-6,\
            l2_regularization_weight=1.0, gaussian_noise_injection_std_dev=0,\
            gradient_clipping_threshold_per_sample=3.0)

training_config = {
    'logdir'            : 'logs', # logdir for log outputs and tensorboard
    'tensorboard_freq'  : 100, # tensorboard record frequence
    'log_freq'          : 200,     # in minibatchs
    'save_freq':3, # save checkpoint frequency
    'train_data'        : 'train.ctf',  # or 'train.tsv'
    'val_data'          : 'dev.ctf',
    'val_interval'      : 3,       # interval in epochs to run validation
    'stop_after'        : 2,       # num epochs to stop if no CV improvement
    'minibatch_seqs'    : 24,      # num sequences of minibatch, when using tsv reader, per worker
    'distributed_after' : 0,       # num sequences after which to start distributed training
    'gpu_pad'           : 0,
    'gpu_cnt'           : 1, # number of gpus
    'multi_gpu'         : False, # using multi GPU training
    # training hyperparameters
    'minibatch_size'    : 15000,    # in samples when using ctf reader, per worker
    'epoch_size'        : 20000,   # in sequences, when using ctf reader
    'max_epochs'        : 100,
    'lr'                : 2,
    'decay':{'epoch':30, 'rate':0.8},
    'learner_handle': create_learner
    }
