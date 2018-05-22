import cntk as C
import numpy as np
from polymath import BiDAF, BiElmo, BiFeature, BiSAF1, BiSAF2
from rnetmodel import RNet, RNetFeature, RNetElmo
from squad_utils import metric_max_over_ground_truths, f1_score, exact_match_score
from helpers import print_para_info, argument_by_name, get_input_variables
import tsv2ctf
import os
import argparse
import importlib
import time
import json, pickle

def create_mb_and_map(input_phs, data_file, polymath, randomize=True, repeat=True):
    '''
    @input_phs dict {'name':input placeholder}
    @data_file str
    @polymath model instance
    '''
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                context_g_words  = C.io.StreamDef('cgw', shape=polymath.wg_dim,     is_sparse=True),
                query_g_words    = C.io.StreamDef('qgw', shape=polymath.wg_dim,     is_sparse=True),
                context_ng_words = C.io.StreamDef('cnw', shape=polymath.wn_dim,     is_sparse=True),
                query_ng_words   = C.io.StreamDef('qnw', shape=polymath.wn_dim,     is_sparse=True),
                answer_begin     = C.io.StreamDef('ab',  shape=polymath.a_dim,      is_sparse=False),
                answer_end       = C.io.StreamDef('ae',  shape=polymath.a_dim,      is_sparse=False),
                context_chars    = C.io.StreamDef('cc',  shape=polymath.word_size,  is_sparse=False),
                query_chars      = C.io.StreamDef('qc',  shape=polymath.word_size,  is_sparse=False),
                query_feature = C.io.StreamDef('qf', shape=1, is_sparse=False),
                doc_feature = C.io.StreamDef('df',shape=3, is_sparse=False) 
                )),
        randomize=randomize,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

    input_map = {
        input_phs['cgw']: mb_source.streams.context_g_words,
        input_phs['qgw']: mb_source.streams.query_g_words,
        input_phs['cnw']: mb_source.streams.context_ng_words,
        input_phs['qnw']: mb_source.streams.query_ng_words,
        input_phs['cc']: mb_source.streams.context_chars,
        input_phs['qc']: mb_source.streams.query_chars,
        input_phs['ab']: mb_source.streams.answer_begin,
        input_phs['ae']: mb_source.streams.answer_end,
    }
    if input_phs.get('qf',None) is not None:
        input_map[input_phs['qf']] = mb_source.streams.query_feature
    if input_phs.get('df',None) is not None:
        input_map[input_phs['df']] = mb_source.streams.doc_feature
    return mb_source, input_map

def create_tsv_reader(input_phs, tsv_file, polymath, seqs, num_workers, is_test=False, misc=None):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        batch_count = 0
        while not(eof and (batch_count % num_workers) == 0):
            batch_count += 1
            batch={'cwids':[], 'qwids':[], 'baidx':[], 'eaidx':[], 'ccids':[], 'qcids':[], 'qf':[],'df':[]}

            while not eof and len(batch['cwids']) < seqs:
                line = f.readline()
                if not line:
                    eof = True
                    break
                '''
                if misc is not None:
                    import re
                    misc['uid'].append(re.match('^([^\t]*)', line).groups()[0])
                '''
                ctokens, qtokens, atokens, cwids, qwids,  baidx, eaidx, ccids, qcids, qf, df \
                    = tsv2ctf.tsv_iter(line, polymath.vocab, polymath.chars, is_test, misc)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['baidx'].append(baidx)
                batch['eaidx'].append(eaidx)
                batch['ccids'].append(ccids)
                batch['qcids'].append(qcids)
                batch['qf'].append(qf.copy())
                batch['df'].append(df.copy())

            if len(batch['cwids']) > 0:
                context_g_words  = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']], polymath.wg_dim)
                context_ng_words = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in batch['cwids']], polymath.wn_dim)
                query_g_words    = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']], polymath.wg_dim)
                query_ng_words   = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in batch['qwids']], polymath.wn_dim)
                context_chars = [np.asarray([[[c for c in cc+[0]*max(0,polymath.word_size-len(cc))]] for cc in ccid], dtype=np.float32) for ccid in batch['ccids']]
                query_chars   = [np.asarray([[[c for c in qc+[0]*max(0,polymath.word_size-len(qc))]] for qc in qcid], dtype=np.float32) for qcid in batch['qcids']]
                answer_begin = [np.asarray(ab, dtype=np.float32) for ab in batch['baidx']]
                answer_end   = [np.asarray(ae, dtype=np.float32) for ae in batch['eaidx']]

                input_map = {input_phs['cgw']:context_g_words,
                         input_phs['qgw']:query_g_words,
                         input_phs['cnw']:context_ng_words,
                         input_phs['qnw']:query_ng_words,
                         input_phs['cc']:context_chars,
                         input_phs['qc']:query_chars,
                         input_phs['ab']:answer_begin,
                         input_phs['ae']:answer_end}
                if input_phs.get('qf',None) is not None:
                    input_map[input_phs['qf']] = batch['qf']
                if input_phs.get('df',None) is not None:
                    input_map[input_phs['df']] = batch['df']
                yield input_map
            else:
                break
    # yield {} # need to generate empty batch for distributed training
from pprint import pprint
def train(data_path, model_path, log_file, config_file, model_name, net, restore=False, profiling=False, gen_heartbeat=False, gpu=0):
    training_config = importlib.import_module(config_file).training_config
    # config for using multi GPUs
    if training_config['multi_gpu']:
        gpu_pad = training_config['gpu_pad']
        gpu_cnt = training_config['gpu_cnt']
        my_rank = C.Communicator.rank()
        my_gpu_id = (my_rank+gpu_pad)%gpu_cnt
        print("rank = "+str(my_rank)+", using gpu "+str(my_gpu_id)+" of "+str(gpu_cnt))
        C.try_set_default_device(C.gpu(my_gpu_id))
    else:
        C.try_set_default_device(C.gpu(gpu))

    # directories
    normal_log = os.path.join(data_path,training_config['logdir'],log_file)
    tensorboard_logdir = os.path.join(data_path,training_config['logdir'],log_file)
    train_data_file = os.path.join(data_path, training_config['train_data'])
    train_data_ext = os.path.splitext(train_data_file)[-1].lower()
    model_file = os.path.join(model_path, model_name)
    print(model_file)
    # record
    epoch_stat = {
        'best_val_err' : 100,
        'best_since'   : 0,
        'val_since'    : 0,
        'record_num'   : 0,
        'epoch':0}

    # training setting
    polymath = choose_model(config_file, net)
    if restore and os.path.isfile(model_file):
        print('reload model {}'.format(model_file))
        polymath.set_model(C.load_model(model_file))
        z = polymath.model
        loss = polymath.loss
        input_phs = polymath.input_phs
        model = z
        #after restore always re-evaluate
        epoch_stat['best_val_err'] = validate_model(os.path.join(data_path, training_config['val_data']), polymath,config_file)
    else:
        z, loss, input_phs = polymath.build_model()
        model = C.combine(list(z.outputs) + [loss.output])

    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']

    progress_writers = [C.logging.ProgressPrinter(
                            num_epochs = max_epochs,
                            freq = log_freq,
                            tag = 'Training',
                            log_to_file = normal_log,
                            rank = C.Communicator.rank(),
                            gen_heartbeat = gen_heartbeat)]
    # add tensorboard writer for visualize
    tensorboard_writer = C.logging.TensorBoardProgressWriter(
                             freq=training_config['tensorboard_freq'],
                             log_dir=tensorboard_logdir,
                             rank = C.Communicator.rank(),
                             model = z)
    progress_writers.append(tensorboard_writer)

    lr_set = training_config['lr']
    if training_config['decay']:
        rate = training_config['decay']['rate']
        epoch = training_config['decay']['epoch']
        lr_set= [(e,(rate**i)*lr_set) for i,e in enumerate(range(1, max_epochs, epoch))]
        print('learning rate set:{}'.format(lr_set))
    lr = C.learning_parameter_schedule(lr_set, minibatch_size=training_config['minibatch_size'], epoch_size=training_config['epoch_size'])

    # learner = C.adadelta(z.parameters, lr, 0.95, 1e-6)
    learner = training_config['learner_handle'](z.parameters, lr)
    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner)

    trainer = C.Trainer(z, (loss, None), learner, progress_writers)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)


    def post_epoch_work(epoch_stat):
        epoch_stat['val_since'] += 1

        if epoch_stat['val_since'] == training_config['val_interval']:
            epoch_stat['val_since'] = 0
            val_err = validate_model(os.path.join(data_path, training_config['val_data']),polymath,config_file)
            if epoch_stat['best_val_err'] > val_err:
                epoch_stat['best_val_err'] = val_err
                epoch_stat['best_since'] = 0
            else:
                epoch_stat['best_since'] += 1
                if epoch_stat['best_since'] > training_config['stop_after']:
                    return False

        if profiling:
            C.debugging.enable_profiler()

        return True

    if train_data_ext == '.ctf':
        mb_source, input_map = create_mb_and_map(input_phs, train_data_file, polymath)

        minibatch_size = training_config['minibatch_size'] # number of samples
        epoch_size = training_config['epoch_size']

        for epoch in range(max_epochs):
            num_seq = 0
            while True:
                if trainer.total_number_of_samples_seen >= training_config['distributed_after']:
                    data = mb_source.next_minibatch(minibatch_size*C.Communicator.num_workers(), input_map=input_map, num_data_partitions=C.Communicator.num_workers(), partition_index=C.Communicator.rank())
                else:
                    data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
                trainer.train_minibatch(data)
                num_seq += trainer.previous_minibatch_sample_count
                if num_seq >= epoch_size:
                    break
            trainer.summarize_training_progress()
            if epoch+1 % training_config['save_freq']==0:
                save_name = os.path.join(model_path,'{}_{}'.format(model_name,epoch))
                print('[TRAIN] save checkpoint into {}'.format(save_name))
                save_flag = True
                while save_flag:
                    os.system('ls -la  >> log.log')
                    os.system('ls -la ./output/models >> log.log')
                    try:
                        trainer.save_checkpoint(save_name)
                        save_flag = False
                    except:
                        print('IO error: try to save model again!')
                        save_flag = True
            if not post_epoch_work(epoch_stat):
                epoch_stat['epoch'] = epoch
                break
    else:
        if train_data_ext != '.tsv':
            raise Exception("Unsupported format")

        minibatch_seqs = training_config['minibatch_seqs'] # number of sequences

        for epoch in range(max_epochs):       # loop over epochs
            tsv_reader = create_tsv_reader(input_phs, train_data_file, polymath, minibatch_seqs, C.Communicator.num_workers())
            minibatch_count = 0
            for data in tsv_reader:
                if (minibatch_count % C.Communicator.num_workers()) == C.Communicator.rank():
                    trainer.train_minibatch(data) # update model with it
                minibatch_count += 1
            trainer.summarize_training_progress()
            if epoch % training_config['save_freq']==0:
                print('[TRAIN] save checkpoint into {}'.format(save_name))
                save_name = os.path.join(model_path,'{}_{}'.format(model_name,epoch))
                os.system('ls -al')
                trainer.save_checkpoint(save_name)
            if not post_epoch_work(epoch_stat):
                epoch_stat['epoch'] = epoch
                break

    if profiling:
        C.debugging.stop_profiler()
    
    print('[TRAIN] training finish after {} epochs'.format(epoch_stat['epoch']))
    save_name = os.path.join(model_path, model_name.split('_')[0])
    print('[TRAIN] save final model as {}.model'.format(save_name))
    model.save(model_file+'.model')

def symbolic_best_span(begin, end):
    running_max_begin = C.layers.Recurrence(C.element_max, initial_state=-float("inf"))(begin)
    return C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30))(running_max_begin + end)

def validate_model(test_data, polymath,config_file):
    print("start validate")
    model = polymath.model
    begin_logits = model.outputs[0]
    end_logits   = model.outputs[1]
    loss         = polymath.loss 
    model = C.combine(begin_logits, end_logits, loss)
    input_phs = polymath.input_phs
    mb_source, input_map = create_mb_and_map(input_phs, test_data, polymath, randomize=False, repeat=False)
    begin_label = input_phs['ab']
    end_label   = input_phs['ae']

    # input placeholder of 2 usage
    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True)

    # max position has gradient 1
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    # mark span with 1 sequence:[0000111111100000]
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
    true_span = C.layers.Recurrence(C.plus)(begin_label - C.sequence.past_value(end_label))
    common_span = C.element_min(predicted_span, true_span)
    # if match
    begin_match = C.sequence.reduce_sum(C.element_min(begin_prediction, begin_label))
    end_match = C.sequence.reduce_sum(C.element_min(end_prediction, end_label))

    predicted_len = C.sequence.reduce_sum(predicted_span)
    true_len = C.sequence.reduce_sum(true_span)
    common_len = C.sequence.reduce_sum(common_span)
    f1 = 2*common_len/(predicted_len+true_len)
    exact_match = C.element_min(begin_match, end_match)
    precision = common_len/predicted_len
    recall = common_len/true_len
    overlap = C.greater(common_len, 0)
    s = lambda x: C.reduce_sum(x, axis=C.Axis.all_axes())
    stats = C.splice(s(f1), s(exact_match), s(precision), s(recall), s(overlap), s(begin_match), s(end_match))

    training_config = importlib.import_module(config_file).training_config
    # Evaluation parameters
    minibatch_size = 100
    num_sequences = 0

    stat_sum = 0
    loss_sum = 0

    while True:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (begin_label in data) or data[begin_label].num_sequences == 0:
            break
        if num_sequences==0: # save attention weight
            save_info(polymath, data)
        out = model.eval(data, outputs=[begin_logits,end_logits,loss], as_numpy=False)
        testloss = out[loss]
        g = best_span_score.grad({begin_prediction:out[begin_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
        other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction], begin_label: data[begin_label], end_label: data[end_label]}
        stat_sum += stats.eval((other_input_map))
        loss_sum += np.sum(testloss.asarray())
        num_sequences += data[begin_label].num_sequences

    stat_avg = stat_sum / num_sequences
    loss_avg = loss_sum / num_sequences

    print("Validated {} sequences, loss {:.4f}, F1 {:.4f}, EM {:.4f}, precision {:4f}, recall {:4f} hasOverlap {:4f}, start_match {:4f}, end_match {:4f}".format(
            num_sequences,
            loss_avg,
            stat_avg[0],
            stat_avg[1],
            stat_avg[2],
            stat_avg[3],
            stat_avg[4],
            stat_avg[5],
            stat_avg[6]))

    return loss_avg
def save_info(polymath, data):
    info = getattr(polymath, 'info',None)
    weights = []
    query_ind = []
    doc_ind = []
    if info is not None:
        for k, v in info.items():
            if k=='query':
                q = v.eval(data) # list(array(*))
                for qq in q:
                    _,indx = np.nonzero(qq)
                    query_ind.append(indx.copy())
            elif k=='doc':
                d = v.eval(data)
                for dd in d:
                    _,indx = np.nonzero(dd)
                    doc_ind.append(indx.copy())
            else:
                res = v.eval(data) # [array for samples]
                weights.append(res) # many kinds of weights 
        save_flag = True
        while save_flag:
            os.system('ls -la  >> log.log')
            os.system('ls -la ./output/visual >> log.log')
            save_name = os.path.join('output','visual',\
                time.strftime("attn_%H%M%S%d", time.localtime()))
            print('[VALIDATION] save weight into {}'.format(save_name))
            with open(save_name, 'wb') as f:
                pickle.dump((query_ind, doc_ind, weights),f)
            save_flag = False
    else:
        print('[FUNCTION]save_info: None info to save')
# map from token to char offset
def w2c_map(s, words):
    w2c=[]
    rem=s
    offset=0
    for i,w in enumerate(words):
        cidx=rem.find(w)
        assert(cidx>=0)
        w2c.append(cidx+offset)
        offset+=cidx + len(w)
        rem=rem[cidx + len(w):]
    return w2c

# get phrase from string based on tokens and their offsets
def get_answer(raw_text, tokens, start, end):
    try:
        w2c=w2c_map(raw_text, tokens)
        return raw_text[w2c[start]:w2c[end]+len(tokens[end])]
    except:
        import pdb
        pdb.set_trace()
def test(test_data, model_path, model_file, config_file, net, gpu=0):
    training_config = importlib.import_module(config_file).training_config
    # config for using multi GPUs
    if training_config['multi_gpu']:
        gpu_pad = training_config['gpu_pad']
        gpu_cnt = training_config['gpu_cnt']
        my_rank = C.Communicator.rank()
        my_gpu_id = (my_rank+gpu_pad)%gpu_cnt
        print("rank = "+str(my_rank)+", using gpu "+str(my_gpu_id)+" of "+str(gpu_cnt))
        C.try_set_default_device(C.gpu(my_gpu_id))
    else:
        C.try_set_default_device(C.gpu(gpu))

    polymath = choose_model(config_file, net)
    model = C.load_model(os.path.join(model_path, model_file))
    begin_logits = model.outputs[0]
    end_logits   = model.outputs[1]
    loss         = C.as_composite(model.outputs[2].owner)
    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_logits.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_logits.dynamic_axes[1], needs_gradient=True)
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))

    batch_size = 5 # in sequences
    misc = {'rawctx':[], 'ctoken':[], 'answer':[], 'uid':[]}
    input_phs = get_input_variables(loss) # TODO check if this is consistent with test
    tsv_reader = create_tsv_reader(input_phs, test_data, polymath, batch_size, 1, is_test=True, misc=misc)
    results = {}
    with open('{}_out.json'.format(model_file), 'w', encoding='utf-8') as json_output:
        for data in tsv_reader:
            out = model.eval(data, outputs=[begin_logits,end_logits,loss], as_numpy=True)
            g = best_span_score.grad({begin_prediction:out[begin_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
            other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction]}
            span = predicted_span.eval((other_input_map))
            for seq, (raw_text, ctokens, answer, uid) in enumerate(zip(misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid'])):
                seq_where = np.argwhere(span[seq])[:,0]
                span_begin = np.min(seq_where)
                span_end = np.max(seq_where)
                predict_answer = get_answer(raw_text, ctokens, span_begin, span_end)
                results['query_id'] = int(uid)
                results['answers'] = [predict_answer]
                results['score'] = float(np.max(out[begin_logits][seq]))+float(np.max(out[end_logits][seq]))
                print('[ANSWER] {} {}'.format(results['query_id'], results['answers']))
                json.dump(results, json_output)
                json_output.write("\n")
            misc['rawctx'] = []
            misc['ctoken'] = []
            misc['answer'] = []
            misc['uid'] = []

def choose_model(config_file,net):
    if net=='BiDAF':
        polymath = BiDAF(config_file)
    if net=='RNet':
        polymath = RNet(config_file)
    if net=='RNetFeature':
        polymath = RNetFeature(config_file)
    if net=='RNetElmo':
        polymath = RNetElmo(config_file)
    if net=='BiFeature':
        polymath = BiFeature(config_file)
    if net=='BiElmo':
        polymath = BiElmo(config_file)
    if net=='BiSAF1':
        polymath = BiSAF1(config_file)
    if net=='BiSAF2':
        polymath = BiSAF2(config_file)
    return polymath
if __name__=='__main__':
    # default Paths relative to current python file.
    abs_path   = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(abs_path, '.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default='output')
    parser.add_argument('-logfile', '--logfile', help='Log file prefix', required=False, default='default')
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
    parser.add_argument('-genheartbeat', '--genheartbeat', help="Turn on heart-beat for philly", action='store_true', default=False)
    parser.add_argument('-config', '--config', help='Config file', required=False, default='config')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-test', '--test', help='Test data file', required=False, default=None)
    parser.add_argument('-model', '--model', help='Model file name, also used for saving', required=False, default='default')
    parser.add_argument('-gpu','--gpu', help='designate which gpu to use', type=int, default=0)
    parser.add_argument('-net', '--net', help='use chosen network model', required=False, default='BiDAF',
                        choices=['BiDAF','RNet','RNetFeature','RNetElmo','BiElmo', 'BiSAF1','BiSAF2','BiFeature'])
    args = vars(parser.parse_args())
    model_path = os.path.join(args['outputdir'],"models")
    if args['datadir'] is not None:
        data_path = args['datadir']

    test_data = args['test']
    test_model = args['model']
    if test_data:
        test(test_data, model_path, test_model, args['config'], args['net'],args['gpu'])
    else:
        try:
            train(data_path, model_path, args['logfile'], args['config'],
                restore = args['restart'], model_name = test_model,
                profiling = args['profile'],
                gen_heartbeat = args['genheartbeat'], gpu=args['gpu'],net=args['net'])
        finally:
            C.Communicator.finalize()
