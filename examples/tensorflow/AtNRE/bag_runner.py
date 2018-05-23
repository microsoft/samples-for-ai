import numpy as np
import time
from nyt_miml_loader import miml_loader as mimlLoader
from bag_loader import loader as Loader
from bag_model import BAGRNN_Model as Model
from bag_trainer import BagTrainer as Trainer

####
# arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='model', help='name of the model')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs in this run')
parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dict', type=str)
parser.add_argument('--embed', type=str, help='embedding pickle file')
parser.add_argument('--model_dir', type=str,default='./model')
parser.add_argument('--log', type=str, default='./log')
parser.add_argument('--eval_dir', type=str, default='./stats')
parser.add_argument('--vocab_size', type=int, default=80000)
parser.add_argument('--bag_num',type=int,default=50, help='number of bags per batch')
parser.add_argument('--L',type=int, default=120, help='maximum sentence length')
parser.add_argument('--embed_dim',type=int)
parser.add_argument('--entity_dim',type=int, default=3, help='embedding dimension for entity position')
parser.add_argument('--enc_dim',type=int,default=512, help='rnn cell size')
parser.add_argument('--cat_n',type=int,default=5, help='number of relations (including NA, NA is cat<0>)')
parser.add_argument('--cell_type',choices=['lstm','gru','pcnn'],default='gru')
parser.add_argument('--dropout', type=float, help='dropout rate')
parser.add_argument('--no-embed-dropout', dest='dropout_embed', action='store_false',
                    help='When specified, dropout will not be performed on the embedding')
parser.set_defaults(dropout_embed=True)
parser.add_argument('--lrate_decay',type=int,default=0,help='number of batches to decay the learning rate by 0.9998')
parser.add_argument('--report_rate',type=float,default=0.5)
parser.add_argument('--test_split',type=int,default=1000)
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--no-DS-data', dest='dsdata', action='store_false')
parser.set_defaults(dsdata=True)
# eps for adversarial training
parser.add_argument('--adv_eps', type=float, help='when specified, adversarial training will be applied')
parser.add_argument('--adv_type', choices=['batch', 'bag', 'sent'], default='sent',
                    help='Type of perturbation normalization: batch-level; bag-level; sent-level;')
parser.add_argument('--adv-only-pos-rel', dest='perturb_all', action='store_false',
                    help='when specified, adversarial training will be applied')
parser.set_defaults(perturb_all=True)
parser.add_argument('--clip_grad', type=float)  # gradient clipping
parser.add_argument('--tune_embed', dest='tune_embed', action='store_true')  # whether fine tune embedding
parser.set_defaults(tune_embed=False)
parser.add_argument('--gpu_usage', type=float, default=0.9)
parser.add_argument('--dataset', choices=['naacl','nyt'], default='naacl')
# top rels considerred in the evaluation
parser.add_argument('--max_eval_rel', type=int, help='number of predicted relations for evaluation, if None, use all')
parser.add_argument('--sampled_sigmoid_loss', type=int,
                    help='if specified, use sampled\weighted sigmoid loss. When >0, number of sampled rels; ' +
                         'When < 0, indicates negative coefficient of weighted sigmoid loss')
parser.add_argument('--include-NA-loss', dest='excl_na', action='store_false',
                    help='if specified, include rel NA in the loss, default excluded (only for sigmoid loss)')
parser.set_defaults(excl_na=True)
parser.add_argument('--softmax_loss', dest='softmax_loss', action='store_true',
                    help='if specified, use softmax loss; default is sigmoid loss')
parser.set_defaults(softmax_loss=False)
# 0, single softmax, >0, full softmax for every rel
parser.add_argument('--softmax_loss_size', type=int, default=0,
                    help='only effect when --softmax_loss. \
                    when == 0, use a single softmax for multi-relation prediction. \
                    when > 0, use a shared softmax layer for each relation and use the loss of positive rels')
parser.add_argument('--max_dist_embed', type=int,
                    help='if specified, use relative distance embedding; otherwise, use one-hot indicator for entities')
parser.add_argument('--warmstart', type=str)
args = parser.parse_args()
###

print('Load Data ...')
ts = time.time()
if args.dataset == 'naacl':
    loader = Loader(relation_file = './data/relation_dict.pkl',
                    label_data_file = ['./data/label_random.pkl', './data/label_gabor.pkl'],
                    unlabel_data_file = './data/DS_noise.pkl',
                    group_eval_data_file = './data/slim_test_group.pkl',
                    embed_dir = args.embed,
                    word_dir = args.dict,
                    n_vocab = args.vocab_size,
                    valid_split = args.test_split,
                    max_len = args.L,
                    use_DS_data = args.dsdata)
else:  # nyt dataset
    loader = mimlLoader(relation_file = './pkl_data/nyt_orig_rel.pkl',
                        train_file = './pkl_data/nyt_orig_train.pkl',
                        test_file='./pkl_data/nyt_orig_test.pkl',
                        embed_dir = './pkl_data/nyt_comb_embed_50d.pkl',
                        n_vocab = args.vocab_size,
                        max_len = args.L)

loader.init_data(bag_batch = args.bag_num,
                 seed = args.seed)

print('>>>>>> done! elapsed = {}'.format(time.time()-ts))

if hasattr(loader, 'embed_dim'):
    embed_dim = loader.embed_dim
else:
    embed_dim = args.embed_dim

print('embed dim = {}'.format(embed_dim))

print('Building Model ...')
ts = time.time()
model = Model(bag_num = args.bag_num,
              enc_dim = args.enc_dim,
              embed_dim = embed_dim,
              # rel_dim = args.enc_dim,
              cat_n = args.cat_n,
              sent_len = args.L,
              word_n = args.vocab_size,
              word_embed = loader.embed,
              dropout = args.dropout,
              cell_type = args.cell_type,
              adv_eps = args.adv_eps,
              adv_type = args.adv_type,
              tune_embed = args.tune_embed,
              use_softmax_loss = args.softmax_loss_size if args.softmax_loss else None,
              sampled_sigmoid_loss = (args.sampled_sigmoid_loss is not None),
              max_dist_embed = args.max_dist_embed,
              excl_na_loss = args.excl_na,
              only_perturb_pos_rel = (not args.perturb_all))

trainer = Trainer(model, loader,
                  lrate = args.lrate,
                  clip_grad = args.clip_grad,
                  lrate_decay_step = args.lrate_decay,
                  sampled_loss = args.sampled_sigmoid_loss,
                  adv_eps = args.adv_eps)

model.build(trainer.is_training, ent_dim = args.entity_dim,
            dropout_embed = args.dropout_embed)

print('>>>>>> done! elapsed = {}'.format(time.time()-ts))

print('Training ...')
restore_dir = args.warmstart
np.random.seed(args.seed)
trainer.train(name = args.name,
              epochs = args.epoch,
              log_dir = args.log,
              model_dir = args.model_dir,
              stats_dir = args.eval_dir,
              restore_dir = restore_dir,
              report_rate = args.report_rate,
              gpu_usage = args.gpu_usage,
              max_eval_rel = args.max_eval_rel)
