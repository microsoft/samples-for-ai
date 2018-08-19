import cntk as C
from cntk.layers import *
from helpers import *
import polymath
import importlib
from convert_elmo import ElmoEmbedder
# =============== factory function ==============
def create_birnn(runit_forward,runit_backward, name=''):
    with C.layers.default_options(initial_state=0.1):
        negRnn = C.layers.Recurrence(runit_backward, go_backwards=True)
        posRnn = C.layers.Recurrence(runit_forward, go_backwards=False)
    @C.Function
    def BiRnn(e):
        h = C.splice(posRnn(e), negRnn(e), name=name)
        return h
    return BiRnn
# ============== class =================
class RNet(polymath.PolyMath):
    def __init__(self, config_file):
        super(RNet,self).__init__(config_file)
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.convs = model_config['char_convs']
        self.highway_layers = model_config['highway_layers']
        self.attn_configs = model_config['attn_configs']
        self.info={}
    def charcnn(self, x):
        conv_out = C.layers.Sequential([
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

        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        word_embed = self.word_glove()(input_glove_words, input_nonglove_words)
        char_embed = self.char_glove()(input_chars)
        embedded = C.splice(word_embed, C.reshape(self.charcnn(char_embed),self.convs), name='splice_embeded')
        
        highway = HighwayNetwork(dim=self.word_emb_dim+self.convs, highway_layers=self.highway_layers)(embedded)
        highway_drop = C.layers.Dropout(self.dropout)(highway)
        processed = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='input_rnn')(highway_drop)

        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')
    def gate_attention_layer(self, inputs, memory, common_len, att_kind='simi'):
        # [#,c][2*d] [#,c][*=q,1]
        if att_kind=='dot':
            qc_attn, attn_weight = self.dot_attention(inputs, memory, common_len).outputs
        else:
            qc_attn, attn_weight = self.simi_attention(inputs, memory).outputs
        inputs = inputs[:common_len]
        qc_attn = qc_attn[:common_len]
        cont_attn = C.splice(inputs, qc_attn) # [#,c][4*d]

        dense = Dropout(self.dropout) >> Dense(2*common_len, activation=C.sigmoid, input_rank=1) >> Label('gate')
        gate = dense(cont_attn) # [#, c][4*d]
        return gate*cont_attn, attn_weight
    def reasoning_layer(self, inputs):
        input_ph = C.placeholder()
        rnn = create_birnn(GRU(self.hidden_dim), GRU(self.hidden_dim),'reasoning_gru')
        block = Sequential([
                LayerNormalization(name='layerbn'), Dropout(self.dropout), rnn
            ])
        res = block(input_ph)
        return C.as_block(
            res,[(input_ph, inputs)], 'reasoning layer', 'reasoning layer'
        )
    def weighted_sum(self, inputs):
        input_ph = C.placeholder()
        weight = Sequential([
            BatchNormalization(),
            Dropout(self.dropout), Dense(self.hidden_dim, activation=C.tanh),
            Dense(1,bias=False),
            C.sequence.softmax
        ])(input_ph) # [#,c][1]
        res = C.sequence.reduce_sum(weight*input_ph)
        return C.as_block(C.combine(res, weight),
            [(input_ph, inputs)], 'weighted sum','weighted sum')
    def output_layer(self, init, memory, inplen):

        def pointer(inputs, state):
            input_ph = C.placeholder()
            state_ph = C.placeholder()
            state_expand = C.sequence.broadcast_as(state_ph, input_ph)
            weight = Sequential([
                BatchNormalization(),
                Dropout(self.dropout),
                Dense(self.hidden_dim, activation=C.sigmoid),Dense(1,bias=False),
                C.sequence.softmax
            ])(C.splice(input_ph, state_expand))
            res = C.sequence.reduce_sum(weight*input_ph)
            return C.as_block(
                C.combine(res, weight),
                [(input_ph, inputs), (state_ph, state)],
                'pointer', 'pointer')
        
        gru = GRU(inplen)
        inp, logits1 = pointer(memory, init).outputs
        state2 = gru(init, inp)
        logits2 = pointer(memory, state2).outputs[1]

        return logits1, logits2 
    def build_model(self):
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
        seif.info['query'] = C.splice(qgw, qnw)
        self.info['doc'] = C.splice(cgw, gnw)
        # graph
        pu, qu = self.input_layer(cgw, cnw, cc, qgw, qnw, qc).outputs
        gate_pu, wei1 = self.gate_attention_layer(pu, qu, common_len=2*self.hidden_dim,attn_kind=self.attn_configs[0]) # [#,c][4*hidden]
        self.info['attn1'] = wei1*1.0
        print('[RNet build]gate_pu:{}'.format(gate_pu))
        pv = self.reasoning_layer(gate_pu) # [#,c][2*hidden]
        gate_self, wei2 = self.gate_attention_layer(pv,pv, common_len=2*self.hidden_dim, att_kind=self.attn_configs[1]) # [#,c][4*hidden]
        self.info['attn2'] = wei2*1.0
        ph = self.reasoning_layer(gate_self) # [#,c][2*hidden]
        init_pu = self.weighted_sum(pu)
        
        start_logits, end_logits  = self.output_layer(init_pu.outputs[0], ph, 2*self.hidden_dim) # [#, c][1]
        
        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        # paper_loss = start_loss + end_loss
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
class RNetFeature(RNet):
    def __init__(self, config_file):
        super(RNetFeature, self).__init__(config_file)
    def build_model(self):
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
                     'qf':qf, 'df':df}
        self._input_phs = input_phs
        self.info['query'] = C.splice(qgw, qnw)
        self.info['doc'] = C.splice(cgw, cnw)
        # graph
        pu, qu = self.input_layer(cgw, cnw, cc, qgw, qnw, qc).outputs
        enhance_pu = C.splice(pu,df); enhance_qu = C.splice(qu, qf)
        gate_pu, wei1 = self.gate_attention_layer(enhance_pu, enhance_qu, common_len=2*self.hidden_dim,\
            att_kind=self.attn_configs[0]) # [#,c][4*hidden]
        self.info['attn1'] = 1.0*wei1
        pv = self.reasoning_layer(gate_pu) # [#,c][2*hidden]
        # self attention 
        gate_self, wei2 = self.gate_attention_layer(pv,pv,common_len=2*self.hidden_dim, att_kind=self.attn_configs[1]) # [#,c][4*hidden]
        self.info['attn2'] = 1.0*wei2
        ph = self.reasoning_layer(gate_self) # [#,c][2*hidden]
        enhance_ph = C.splice(ph, df)
        init_pu = self.weighted_sum(enhance_pu)

        start_logits, end_logits  = self.output_layer(init_pu.outputs[0], enhance_ph, 2*self.hidden_dim+3) # [#, c][1]
        self.info['start_logits'] = start_logits*1.0
        self.info['end_logits'] = end_logits*1.0
 
        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        # paper_loss = start_loss + end_loss
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
class RNetElmo(RNet):
    def __init__(self, config_file):
        super(RNetElmo, self).__init__(config_file)
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
    def build_model(self):
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
                     'qf':qf, 'df':df}
        self._input_phs = input_phs
        self.info['query'] = C.splice(qgw, qnw)
        self.info['doc'] = C.splice(cgw, cnw)
        # graph
        elmo_encoder = self.__elmo_fac.build()
        #input layer
        reduction_cc = C.reshape(cc,(-1,))
        reduction_qc = C.reshape(qc, (-1,))
        c_elmo = elmo_encoder(reduction_cc)
        q_elmo = elmo_encoder(reduction_qc)
        pu, qu = self.input_layer(cgw, cnw, qgw, qnw).outputs
        enhance_pu = C.splice(pu,c_elmo,df); enhance_qu = C.splice(qu,q_elmo,qf)
        gate_pu, wei1 = self.gate_attention_layer(enhance_pu, enhance_qu, common_len=2*self.hidden_dim+1024,\
            att_kind=self.attn_configs[0]) # [#,c][4*hidden]
        self.info['attn1'] = 1.0*wei1
        pv = self.reasoning_layer(gate_pu) # [#,c][2*hidden]
        # self attention 
        gate_self, wei2 = self.gate_attention_layer(pv, pv,common_len=2*self.hidden_dim,att_kind=self.attn_configs[1]) # [#,c][4*hidden]
        self.info['attn2'] = 1.0*wei2
        ph = self.reasoning_layer(gate_self) # [#,c][2*hidden]
        enhance_ph = C.splice(ph, c_elmo, df)
        init_pu = self.weighted_sum(enhance_pu)

        start_logits, end_logits  = self.output_layer(init_pu.outputs[0], enhance_ph, 2*self.hidden_dim+1027) # [#, c][1]
        self.info['start_logits'] = start_logits*1.0
        self.info['end_logits'] = end_logits*1.0
 
        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        # paper_loss = start_loss + end_loss
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
        
# =============== test edition ==================
from cntk.debugging import debug_model
def test_model_part():
    from train_pm import  create_mb_and_map
    rnet = RNet('config')
    model,loss, input_phs = rnet.build_model()
    mb, input_map = create_mb_and_map(input_phs, 'dev.ctf', rnet)
    data=mb.next_minibatch(3,input_map=input_map)
    res = model.eval(data)
    print(res)
def _testcode():
    data=[np.array([[1,2,3,0],[1,2,3,0]]),
        np.array([[1,2,0,0],[2,3,0,0]]),
        np.array([[4,0,0,0],[5,0,0,0],[6,0,0,0]])]
    inp=C.sequence.input_variable(4)

if __name__ == '__main__':
    C.try_set_default_device(C.gpu(2))
    test_model_part()
