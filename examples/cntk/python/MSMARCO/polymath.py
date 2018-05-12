import cntk as C
import numpy as np
from helpers import *
import pickle
import importlib
import os
from cntk.initializer import glorot_uniform, xavier
from convert_elmo import ElmoEmbedder
from cntk.layers import *

class PolyMath(object):
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.word_count_threshold = data_config['word_count_threshold']
        self.char_count_threshold = data_config['char_count_threshold']
        self.word_size = data_config['word_size']
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, data_config['pickle_file'])
        self.word_embed_file = data_config['word_embed_file']
        self.char_embed_file = data_config['char_embed_file']

        with open(pickle_file, 'rb') as vf:
            known, self.vocab, self.chars = pickle.load(vf)

        self.wg_dim = known
        self.wn_dim = len(self.vocab) - known
        self.c_dim = len(self.chars)
        self.a_dim = 1

        self.hidden_dim = model_config['hidden_dim']
        self.dropout = model_config['dropout']
        self.char_emb_dim = model_config['char_emb_dim']
        self.word_emb_dim = model_config['word_emb_dim']
        self.use_cudnn = model_config['use_cudnn']
        self.use_sparse = False
        # self.ldb = model_config['lambda'] # 控制两个loss的平衡

        print('dropout', self.dropout)
        print('use_cudnn', self.use_cudnn)
        print('use_sparse', self.use_sparse)

        self._model=None
        self._loss=None
        self._input_phs=None
        self._metric = None
        self.info = {} # use to record information 
    def word_glove(self):
        # load glove
        if os.path.isfile('glove300.model'):
            print('[BUILD] load glove300.model')
            return C.load_model('glove300.model')
        npglove = np.zeros((self.wg_dim, self.word_emb_dim), dtype=np.float32)
        with open(os.path.join(self.abs_path, self.word_embed_file), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if self.vocab.get(word, self.wg_dim)<self.wg_dim:
                    npglove[self.vocab[word],:] = np.asarray([float(p) for p in parts[-300:]])
        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(len(self.vocab) - self.wg_dim, self.word_emb_dim), init=C.glorot_uniform(), name='TrainableE')
        @C.Function
        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)
        
        func.save('glove300.model')
        print('[BUILD] save glove300.model')
        return func
    def char_glove(self):
        npglove = np.zeros((self.c_dim, self.char_emb_dim), dtype=np.float32)
        # only 94 known chars, 308 chars in all
        with open(os.path.join(self.abs_path, self.char_embed_file), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if self.chars.get(word, self.c_dim)<self.c_dim:
                    npglove[self.chars[word],:] = np.asarray([float(p) for p in parts[-300:]])
        glove = C.constant(npglove)
        @C.Function
        def func(cg):
             return C.times(cg, glove)
        return func
    def word_level_drop(self, doc):
        # doc [#, c][d]
        seq_shape=C.sequence.is_first(doc)
        u = C.random.uniform_like(seq_shape, seed=98052)
        mask = C.element_select(C.greater(u, 0.08),1.0,0)
        return doc*mask
    def dot_attention(self,inputs, memory, dim):
        '''
        @inputs: [#,c][d] a sequence need attention
        @memory(key): [#,q][d] a sequence input refers to compute similarity(weight)
        @value: [#,q][d] a sequence input refers to weighted sum
        @output: [#,c][d] attention vector
        '''
        input_ph = C.placeholder()
        input_mem = C.placeholder()
        with C.layers.default_options(bias=False, activation=C.relu): # all the projections have no bias
            attn_proj_enc = C.layers.Dense(dim, init=glorot_uniform(), input_rank=1, name="Wqu")
            attn_proj_dec = C.layers.Dense(dim, init=glorot_uniform(), input_rank=1)

        inputs_ = attn_proj_enc(input_ph) # [#,c][d]
        memory_ = attn_proj_dec(input_mem) # [#,q][d]
        unpack_memory, mem_mask = C.sequence.unpack(memory_, 0).outputs # [#][*=q, d], [#][*=q]
        unpack_memory_expand = C.sequence.broadcast_as(unpack_memory, inputs_) # [#,c][*=q,d]

        matrix = C.times_transpose(inputs_, unpack_memory_expand)/(dim**0.5) # [#,c][*=q]
        mem_mask_expand = C.sequence.broadcast_as(mem_mask, inputs_) # [#,c][*=q]
        matrix = C.element_select(mem_mask_expand, matrix, C.constant(-1e+30)) # [#,c][*=q]
        logits = C.reshape(C.softmax(matrix),(-1,1)) # [#,c][*=q,1]
        # [#,c][*=q, d]
        memory_expand = C.sequence.broadcast_as(C.sequence.unpack(input_mem, 0,no_mask_output=True), input_ph)
        weighted_att = C.reshape(C.reduce_sum(logits*memory_expand, axis=0),(-1,)) # [#,c][d]

        return C.as_block(
            C.combine(weighted_att, logits),
            [(input_ph, inputs), (input_mem, memory)],
            'dot attention',
            'dot attention'
        )
    def simi_attention(self, input, memory):
        '''
        return:
        memory weighted vectors over input [#,c][d]
        weight
        '''
        input_ph = C.placeholder() # [#,c][d]
        mem_ph = C.placeholder() # [#,q][d]
        
        input_dense = Dense(2*self.hidden_dim, bias=False,input_rank=1)
        mem_dense = Dense(2*self.hidden_dim, bias=False,input_rank=1)
        bias = C.Parameter(shape=(2*self.hidden_dim,), init=0.0)
        weight_dense = Dense(1,bias=False, input_rank=1)

        proj_inp = input_dense(input_ph) # [#,c][d]
        proj_mem = mem_dense(mem_ph) # [#,q][d]
        unpack_memory, mem_mask = C.sequence.unpack(proj_mem, 0).outputs # [#][*=q, d] [#][*=q]
        expand_mem = C.sequence.broadcast_as(unpack_memory, proj_inp) # [#,c][*=q,d]
        expand_mask = C.sequence.broadcast_as(mem_mask, proj_inp) # [#,c][*=q]
        matrix = C.reshape( weight_dense(C.tanh(proj_inp + expand_mem + bias)) , (-1,)) # [#,c][*=q]
        matrix = C.element_select(expand_mask, matrix, -1e30)
        logits = C.softmax(matrix, axis=0) # [#,c][*=q]
        weight_mem = C.reduce_sum(C.reshape(logits, (-1,1))*expand_mem, axis=0) # [#,c][d]
        weight_mem = C.reshape(weight_mem, (-1,))

        return C.as_block(
            C.combine(weight_mem, logits),
            [(input_ph, input),(mem_ph, memory)],
            'simi_attention','simi_attention'
        )
    def multiHead(self, context, query, outdim, head=4):
        cph=C.placeholder()
        qph=C.placeholder()
        atts = []
        for i in range(head):
            dense_q = C.layers.Dense(outdim, activation=C.relu, init=xavier(1.377),
                bias=False, input_rank=1, name='headq_{}'.format(i))(qph)
            dense_c = C.layers.Dense(outdim, activation=C.relu, init=xavier(1.377),
                bias=False, input_rank=1, name='headc_{}'.format(i))(cph)
            attn,_ = self.dot_attention(dense_c,dense_q,outdim).outputs
            atts.append(attn)
        res = C.splice(*atts)
        return C.as_block(res,[(cph, context),(qph, query)],'multiHead','multiHead')
    def build_model(self):
        raise NotImplementedError
    def set_model(self, model):
        self._model=model
        self._input_phs=get_input_variables(model)
        self._loss=model.outputs[2]
    @property
    def model(self):
        if not self._model:
            self._model, self._loss, self._input_phs = self.build_model()
        return self._model
    @property
    def loss(self):
        if not self._model:
            self._model, self._loss, self._input_phs = self.build_model()
        return self._loss
    @property
    def input_phs(self):
        if not self._model:
            self._model, self._loss, self._input_phs = self.build_model()
        return self._input_phs
    @property
    def metric(self):
        return self._metric
class BiDAF(PolyMath):
    def __init__(self, config_file):
        super(BiDAF, self).__init__(config_file)
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.convs = model_config['char_convs']
        self.highway_layers = model_config['highway_layers']
        self.two_step = model_config['two_step']
    def get_inputs(self):
        if self._input_phs is not None:
            return self._input_phs
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qnw')
        cc = C.input_variable((1,self.word_size), dynamic_axes=[b,c], name='cc')
        qc = C.input_variable((1,self.word_size), dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')
        input_phs = {'cgw':cgw, 'cnw':cnw, 'qgw':qgw, 'qnw':qnw,
                        'cc':cc, 'qc':qc, 'ab':ab, 'ae':ae}
        self._input_phs = input_phs
        return self._input_phs
    def charcnn(self, x):
        conv_out = C.layers.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5,self.char_emb_dim), self.convs, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling
    def input_layer(self,cgw,cnw,cc,qgw,qnw,qc):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        cc_ph  = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph  = C.placeholder()

        input_chars = C.placeholder(shape=(1,self.word_size,self.c_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        # we need to reshape because GlobalMaxPooling/reduce_max is retaining a trailing singleton dimension
        # todo GlobalPooling/reduce_max should have a keepdims default to False
        embedded = C.splice(
            C.reshape(self.charcnn(input_chars), self.convs),
            self.word_glove()(input_glove_words, input_nonglove_words), name='splice_embed')
        highway = HighwayNetwork(dim=self.word_emb_dim+self.convs, highway_layers=self.highway_layers)(embedded)
        highway_drop = C.layers.Dropout(self.dropout)(highway)
        processed = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='input_rnn')(highway_drop)

        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)

        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')
    def attention_layer(self, context, query, dim):
        q_processed = C.placeholder(shape=(dim,))
        c_processed = C.placeholder(shape=(dim,))

        #convert query's sequence axis to static
        qvw, qvw_mask = C.sequence.unpack(q_processed, padding_value=0).outputs

        # This part deserves some explanation
        # It is the attention layer
        # In the paper they use a 6 * dim dimensional vector
        # here we split it in three parts because the different parts
        # participate in very different operations
        # so W * [h; u; h.* u] becomes w1 * h + w2 * u + w3 * (h.*u)
        ws1 = C.parameter(shape=(dim, 1), init=C.glorot_uniform())
        ws2 = C.parameter(shape=(dim, 1), init=C.glorot_uniform())
        ws3 = C.parameter(shape=(1, dim), init=C.glorot_uniform())
        att_bias = C.parameter(shape=(), init=0)

        wh = C.times (c_processed, ws1)
        wu = C.reshape(C.times (qvw, ws2), (-1,))
        whu = C.reshape(C.reduce_sum(c_processed * C.sequence.broadcast_as(qvw * ws3, c_processed), axis=1), (-1,))
        S = wh + whu + C.sequence.broadcast_as(wu, c_processed) + att_bias
        # mask out values outside of Query, and fill in gaps with -1e+30 as neutral value for both reduce_log_sum_exp and reduce_max
        qvw_mask_expanded = C.sequence.broadcast_as(qvw_mask, c_processed)
        S = C.element_select(qvw_mask_expanded, S, C.constant(-1e+30))
        q_attn = C.reshape(C.softmax(S), (-1,1))
        #q_attn = print_node(q_attn)
        c2q = C.reshape(C.reduce_sum(C.sequence.broadcast_as(qvw, q_attn) * q_attn, axis=0),(-1))

        max_col = C.reduce_max(S)
        c_attn = C.sequence.softmax(max_col)

        htilde = C.sequence.reduce_sum(c_processed * c_attn)
        q2c = C.sequence.broadcast_as(htilde, c_processed)
        q2c_out = c_processed * q2c

        att_context = C.splice(c_processed, c2q, c_processed * c2q, q2c_out)
        res = C.combine(att_context, C.reshape(q_attn,(-1,)))
        return C.as_block(
            res,
            [(c_processed, context), (q_processed, query)],
            'attention_layer',
            'attention_layer')
    def modeling_layer(self, attention_context):
        att_context = C.placeholder()
        #modeling layer
        # todo: use dropout in optimized_rnn_stack from cudnn once API exposes it
        mod_context = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn0'),
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn1')])(att_context)

        return C.as_block(
            mod_context,
            [(att_context, attention_context)],
            'modeling_layer',
            'modeling_layer')
    def output_layer(self, attention_context, modeling_context):
        att_context = C.placeholder()
        mod_context = C.placeholder()
        #output layer [#,c][1]
        start_logits = C.layers.Dense(1, name='out_start')(C.dropout(C.splice(mod_context, att_context), self.dropout))
        start_logits = C.sequence.softmax(start_logits)
        start_hardmax = seq_hardmax(start_logits) # [000010000]
        att_mod_ctx = C.sequence.last(C.sequence.gather(mod_context, start_hardmax)) # [#][2*hidden_dim]
        att_mod_ctx_expanded = C.sequence.broadcast_as(att_mod_ctx, att_context)
        end_input = C.splice(att_context, mod_context, att_mod_ctx_expanded, mod_context * att_mod_ctx_expanded) # [#, c][14*hidden_dim]
        m2 = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='output_rnn')(end_input)
        end_logits = C.layers.Dense(1, name='out_end')(C.dropout(C.splice(m2, att_context), self.dropout))
        end_logits = C.sequence.softmax(end_logits)

        return C.as_block(
            C.combine([start_logits, end_logits]),
            [(att_context, attention_context), (mod_context, modeling_context)],
            'output_layer',
            'output_layer')
    def build_model(self):
        phmap = self.get_inputs()        
        ab = phmap['ab']
        ae = phmap['ae']
        #input layer
        c_processed, q_processed = self.input_layer(phmap['cgw'],phmap['cnw'],phmap['cc'],\
            phmap['qgw'],phmap['qnw'],phmap['qc']).outputs

        # attention layer
        att_context, wei = self.attention_layer(c_processed, q_processed,dim=2*self.hidden_dim).outputs

        # modeling layer
        mod_context = self.modeling_layer(att_context)

        # output layer
        start_logits, end_logits = self.output_layer(att_context, mod_context).outputs

        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
class BiFeature(BiDAF):
    def __init__(self, config_file):
        super(BiFeature, self).__init__(config_file)
    def get_inputs(self):
        if self._input_phs is not None:
            return self._input_phs
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qnw')
        cc = C.input_variable((1,self.word_size), dynamic_axes=[b,c], name='cc')
        qc = C.input_variable((1,self.word_size), dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')
        qf = C.input_variable(1, dynamic_axes=[b,q], is_sparse=False, name='query_feature')
        df = C.input_variable(3, dynamic_axes=[b,c], is_sparse=False, name='doc_feature')
        input_phs = {'cgw':cgw, 'cnw':cnw, 'qgw':qgw, 'qnw':qnw,
                     'cc':cc, 'qc':qc, 'ab':ab, 'ae':ae,
                     'df':df, 'qf':qf}
        self._input_phs = input_phs
        return self._input_phs
    def attention_layer(self, context, query, dimc, dimq, common_dim):
        q_processed = C.placeholder(shape=(dimq,))
        c_processed = C.placeholder(shape=(dimc,))

        #convert query's sequence axis to static
        qvw, qvw_mask = C.sequence.unpack(q_processed, padding_value=0).outputs

        # so W * [h; u; h.* u] becomes w1 * h + w2 * u + w4 * (h.*u)
        ws1 = C.parameter(shape=(dimc, 1), init=C.glorot_uniform())
        ws2 = C.parameter(shape=(dimq, 1), init=C.glorot_uniform())
        ws4 = C.parameter(shape=(1, common_dim), init=C.glorot_uniform())
        att_bias = C.parameter(shape=(), init=0)

        wh = C.times (c_processed, ws1) # [#,c][1]
        wu = C.reshape(C.times (qvw, ws2), (-1,)) # [#][*]
        # qvw*ws4: [#][*,200], whu:[#,c][*]
        whu = C.reshape(C.reduce_sum(
            c_processed[:common_dim] *\
            C.sequence.broadcast_as(qvw[:,:common_dim] * ws4, c_processed), axis=1), (-1,))
        S1 = wh + C.sequence.broadcast_as(wu, c_processed) + att_bias # [#,c][*]
        qvw_mask_expanded = C.sequence.broadcast_as(qvw_mask, c_processed)
        S1 = C.element_select(qvw_mask_expanded, S1, C.constant(-1e+30))
        q_attn = C.reshape(C.softmax(S1), (-1,1)) # [#,c][*,1]
        c2q = C.reshape(C.reduce_sum(C.sequence.broadcast_as(qvw, q_attn) * q_attn, axis=0),(-1)) # [#,c][200]

        max_col = C.reduce_max(S1) # [#,c][1] 最大的q中的单词
        c_attn = C.sequence.softmax(max_col) # [#,c][1] 对c中的每一个单词做softmax

        htilde = C.sequence.reduce_sum(c_processed * c_attn) # [#][200]
        q2c = C.sequence.broadcast_as(htilde, c_processed) # [#,c][200]
        q2c_out = c_processed[:common_dim] * q2c[:common_dim]

        # 原始文档，题目表示，文章重点表示，匹配度表示，文章上下文表示
        att_context_reg = C.splice(c_processed, c2q, q2c_out, 
                    c_processed[:common_dim]*c2q[:common_dim])
        res = C.combine(att_context_reg, C.reshape(q_attn,(-1,)))
        return \
        C.as_block(res,
            [(c_processed, context), (q_processed, query)],
            'attention_layer',
            'attention_layer')
    def build_model(self):
        phmap = self.get_inputs()
        df = phmap['df']
        qf = phmap['qf']
        ab = phmap['ab']
        ae = phmap['ae']
        #input layer
        cc = C.reshape(phmap['cc'], (1,-1)); qc = C.reshape(phmap['qc'], (1,-1))
        c_processed, q_processed = self.input_layer(phmap['cgw'],phmap['cnw'],cc,\
            phmap['qgw'],phmap['qnw'],qc).outputs
        c_processed = C.splice(c_processed, df)
        q_processed = C.splice(q_processed, qf)
        # attention layer output:[#,c][8*hidden_dim]
        att_context, wei1 = self.attention_layer(c_processed, q_processed,
            dimc=2*self.hidden_dim+3, dimq=2*self.hidden_dim+1,
            common_dim=2*self.hidden_dim).outputs
        # modeling layer output:[#][1] [#,c][2*hidden_dim]
        mod_context_reg = self.modeling_layer(att_context)
        # output layer
        mod_context_reg = C.splice(mod_context_reg, df)
        start_logits, end_logits = self.output_layer(att_context, mod_context_reg).outputs

        # loss
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        res = C.combine([start_logits, end_logits])
        self._model = res
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
class BiElmo(BiFeature):
    def __init__(self, config_file):
        super(BiElmo, self).__init__(config_file)
        self.__elmo_fac = ElmoEmbedder()
    def input_layer(self,cgw,cnw,qgw,qnw):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()

        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        # we need to reshape because GlobalMaxPooling/reduce_max is retaining a trailing singleton dimension
        # todo GlobalPooling/reduce_max should have a keepdims default to False
        embedded = self.word_glove()(input_glove_words, input_nonglove_words)
        highway = HighwayNetwork(dim=self.word_emb_dim, highway_layers=self.highway_layers)(embedded)
        highway_drop = C.layers.Dropout(self.dropout)(highway)
        processed = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='input_rnn')(highway_drop)

        q_processed = processed.clone(C.CloneMethod.share, {input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(qgw_ph, qgw),(qnw_ph, qnw)],
            'input_layer',
            'input_layer')
    def self_attention_layer(self, context):
        dense = C.layers.Dense(self.hidden_dim, activation=C.relu)
        rnn = OptimizedRnnStack(self.hidden_dim,bidirectional=True, use_cudnn=self.use_cudnn)
        context1 = dense(context)
        process_context = rnn(context1)
        # residual attention
        att_context, wei = self.dot_attention(process_context,process_context,\
            self.hidden_dim).outputs
        dense2 = C.layers.Dense(self.hidden_dim, activation=C.relu)(att_context)
        res = dense2+context1
        return dense2
    def build_model(self):
        phmap = self.get_inputs()
        cc = phmap['cc']
        qc = phmap['qc']
        ab = phmap['ab']
        ae = phmap['ae']
        df = phmap['df']
        qf = phmap['qf']
        #self.info['query'] = C.splice(qgw, qnw)
        #self.info['doc'] = C.splice(cgw, gnw)
        elmo_encoder = self.__elmo_fac.build()
        #input layer
        reduction_cc = C.reshape(cc,(-1,))
        reduction_qc = C.reshape(qc, (-1,))
        c_elmo = elmo_encoder(reduction_cc)
        q_elmo = elmo_encoder(reduction_qc)
        c_processed, q_processed = self.input_layer(phmap['cgw'],phmap['cnw'],phmap['qgw'],phmap['qnw']).outputs

        # attention layer
        c_enhance = C.splice(c_processed, c_elmo, df)
        q_enhance = C.splice(q_processed, q_elmo, qf)
        att_context, wei = self.attention_layer(c_enhance, q_enhance,
            dimc= 2*self.hidden_dim+1027, dimq=2*self.hidden_dim+1025,\
            common_dim=2*self.hidden_dim+1024).outputs
        self_context = self.self_attention_layer(att_context) # 2*hidden_dim
        # modeling layer
        mod_context = self.modeling_layer(self_context)
        enhance_mod_context = C.splice(mod_context, c_elmo, df)

        # output layer
        start_logits, end_logits = self.output_layer(att_context, enhance_mod_context).outputs

        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        regulizer = 0.001*C.reduce_sum(elmo_encoder.scales*elmo_encoder.scales)
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae) + regulizer
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
class BiSAF1(BiFeature):
    def __init__(self, config_file):
        super(BiSAF1, self).__init__(config_file)
    def build_model(self):
        phmap = self.get_inputs()
        df = phmap['df']
        qf = phmap['qf']
        ab = phmap['ab']
        ae = phmap['ae']
        #input layer
        cc = C.reshape(phmap['cc'], (1,-1)); qc = C.reshape(phmap['qc'], (1,-1))
        c_processed, q_processed = self.input_layer(phmap['cgw'],phmap['cnw'],cc,\
            phmap['qgw'],phmap['qnw'],qc).outputs
        c_processed = C.splice(c_processed, df)
        q_processed = C.splice(q_processed, qf)
        self_context = self.multiHead(c_processed, c_processed, self.hidden_dim//2)
        self_context = C.splice(self_context, df)
        # attention layer output:[#,c][8*hidden_dim]
        att_context, wei1 = self.attention_layer(self_context, q_processed, dimc=2*self.hidden_dim+3,
            dimq=2*self.hidden_dim+1,
            common_dim=2*self.hidden_dim).outputs

        # modeling layer
        mod_context = self.modeling_layer(att_context)
        mod_context = C.splice(mod_context, df)

        # output layer
        start_logits, end_logits = self.output_layer(att_context, mod_context).outputs

        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
class BiSAF2(BiFeature):
    def __init__(self, config_file):
        super(BiSAF2, self).__init__(config_file)
    def build_model(self):
        phmap = self.get_inputs()
        df = phmap['df']
        qf = phmap['qf']
        ab = phmap['ab']
        ae = phmap['ae']
        #input layer
        cc = C.reshape(phmap['cc'], (1,-1)); qc = C.reshape(phmap['qc'], (1,-1))
        c_processed, q_processed = self.input_layer(phmap['cgw'],phmap['cnw'],cc,\
            phmap['qgw'],phmap['qnw'],qc).outputs
        c_processed = C.splice(c_processed, df)
        q_processed = C.splice(q_processed, qf)
        # attention layer output:[#,c][8*hidden_dim]
        att_context, wei1 = self.attention_layer(c_processed, q_processed, dimc=2*self.hidden_dim+3,
            dimq=2*self.hidden_dim+1,
            common_dim=2*self.hidden_dim).outputs
        a = att_context[:4*self.hidden_dim]; b = att_context[4*self.hidden_dim:]
        self_context = self.multiHead(a,a,self.hidden_dim//2)

        # modeling layer
        mod_inp = C.splice(self_context, b)
        mod_context = self.modeling_layer(mod_inp)
        mod_context = C.splice(mod_context, df)

        # output layer
        start_logits, end_logits = self.output_layer(att_context, mod_context).outputs

        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
# ====================================
from subnetworks import IndRNN
class BiDAFInd(BiDAF):
    def __init__(self, config_file):
        super(BiDAFInd, self).__init__(config_file)
        model_config = importlib.import_module(config_file).model_config

        self._time_step = 100 #TODO: dataset explore
        self._indrnn_builder = IndRNN(self.hidden_dim, self.hidden_dim, recurrent_max_abs=pow(2, 1/self._time_step), activation = C.leaky_relu)
        self.use_layerbn = model_config['use_layerbn']

    def charcnn(self, x):
        '''
        @x:[I,w1,w2,w3,...]
        @kernal:[O,I,w1,w2,w3,...]
        @out:[O,?depend on stride]
        '''
        conv_out = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5,self.char_emb_dim), self.convs, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

    def input_layer(self, cgw,cnw,cc,qgw,qnw,qc):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        cc_ph  = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph  = C.placeholder()
        input_chars = C.placeholder(shape=(1,self.word_size,self.c_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        word_embed = self.word_glove()(input_glove_words, input_nonglove_words)
        char_embed = self.char_glove()(input_chars)
        embeded = C.splice(word_embed, C.reshape(self.charcnn(char_embed),self.convs), name='splice_embeded')

        self._indrnn_builder._input_size = self.word_emb_dim+self.convs
        ind1 = [self._indrnn_builder.build(), self._indrnn_builder.build()]
        self._indrnn_builder._input_size = 2*self.hidden_dim
        indrnns = [self._indrnn_builder.build() for _ in range(4)]
        indrnns = ind1+indrnns

        process = C.layers.For(
            range(3),lambda i:C.layers.Sequential([
                C.layers.Dropout(self.dropout),
                (C.layers.Recurrence(indrnns[2*i]),C.layers.Recurrence(indrnns[2*i+1],go_backwards=True)),
                C.splice
            ]))
        processed = process(embeded)

        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')

    def modeling_layer(self, attention_context):
        att_context = C.placeholder(shape=(8*self.hidden_dim,))
        self._indrnn_builder._input_size = 8*self.hidden_dim
        ind1 = [self._indrnn_builder.build(), self._indrnn_builder.build()]
        self._indrnn_builder._input_size = 2*self.hidden_dim
        indrnns = [self._indrnn_builder.build() for _ in range(10)]
        indrnns = ind1+indrnns
        #modeling layer 6 resnet layers
        model = C.layers.For(range(3),lambda i:C.layers.Sequential([
            #C.layers.ResNetBlock(
                C.layers.Sequential([
                    C.layers.LayerNormalization() if self.use_layerbn else C.layers.identity,
                    C.layers.Dropout(self.dropout),
                    (C.layers.Recurrence(indrnns[4*i]), C.layers.Recurrence(indrnns[4*i+1],go_backwards=True)),
                    C.splice,
                    C.layers.LayerNormalization() if self.use_layerbn else C.layers.identity,
                    C.layers.Dropout(self.dropout),
                    (C.layers.Recurrence(indrnns[4*i+2]), C.layers.Recurrence(indrnns[4*i+3],go_backwards=True)),
                    C.splice
                ])
            #)
        ]))
        mod_context = model(att_context)
        return C.as_block(
            mod_context,
            [(att_context, attention_context)],
            'modeling_layer',
            'modeling_layer')

class BiDAFCoA(BiDAF):
    def __init__(self, config_file):
       super(BiDAFCoA, self).__init__(config_file)
    def attention_layer(self, context, query,dim):
        input_ph = C.placeholder(shape=(dim,))
        input_mem = C.placeholder(shape=(dim,))
        with C.layers.default_options(bias=False, activation=C.relu):
            attn_proj_enc = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
            attn_proj_dec = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1)

        inputs_ = attn_proj_enc(input_ph) # [#,c][d]
        memory_ = attn_proj_dec(input_mem) # [#,q][d]

        cln_mem_ph = C.placeholder() # [#,q][?=d]
        cln_inp_ph = C.placeholder() # [#,c][?=d]
        unpack_inputs, inputs_mask = C.sequence.unpack(cln_inp_ph, 0).outputs # [#][*=c,d] [#][*=c]
        expand_inputs = C.sequence.broadcast_as(unpack_inputs, cln_mem_ph) # [#,q][*=c,d]
        matrix = C.reshape(C.times_transpose(cln_mem_ph, expand_inputs)/(self.hidden_dim**0.5),(-1,)) # [#,q][*=c]
        matrix = C.element_select(C.sequence.broadcast_as(inputs_mask,cln_mem_ph), matrix, C.constant(-1e30))
        logits = C.softmax(matrix, axis=0, name='level 1 weight') # [#,q][*=c]
        trans_expand_inputs = C.transpose(expand_inputs,[1,0]) # [#,q][d,*=c]
        q_over_c = C.reshape(C.reduce_sum(logits*trans_expand_inputs,axis=1),(-1,))/(self.hidden_dim**0.5) # [#,q][d]
        new_q = C.splice(cln_mem_ph, q_over_c) # [#,q][2*d]
        # over
        unpack_matrix, matrix_mask = C.sequence.unpack(matrix,0).outputs # [#][*=q,*=c] [#][*=q]
        inputs_mask_s = C.to_sequence(C.reshape(inputs_mask,(-1,1))) # [#,c'][1]
        trans_matrix = C.to_sequence_like(C.transpose(unpack_matrix,[1,0]), inputs_mask_s) # [#,c'][*=q]
        trans_matrix = C.sequence.gather(trans_matrix, inputs_mask_s) # [#,c2][*=q]
        trans_matrix = C.element_select(C.sequence.broadcast_as(matrix_mask, trans_matrix), trans_matrix, C.constant(-1e30))
        logits2 = C.softmax(trans_matrix, axis=0, name='level 2 weight') # [#,c2][*=c]
        unpack_new_q, new_q_mask = C.sequence.unpack(new_q,0).outputs # [#][*=q,2*d] [#][*=q]
        expand_new_q = C.transpose(C.sequence.broadcast_as(unpack_new_q, trans_matrix),[1,0]) # [#,c2][2d,*=q]
        c_over_q = C.reshape(C.reduce_sum(logits2*expand_new_q, axis=1),(-1,))/(2*self.hidden_dim)**0.5 # [#,c2][2d]
        c_over_q = C.reconcile_dynamic_axes(c_over_q, cln_inp_ph)

        weighted_q = c_over_q.clone(C.CloneMethod.share, {cln_mem_ph: memory_, cln_inp_ph: inputs_}) # [#,c][2d]
        c2c = q_over_c.clone(C.CloneMethod.share, {cln_mem_ph: inputs_, cln_inp_ph: inputs_}) # [#,c][2d]

        att_context = C.splice(input_ph, weighted_q, c2c) # 2d+2d+2d

        return C.as_block(
            att_context,
            [(input_ph, context),(input_mem, query)],
            'attention_layer','attention_layer'
        )

