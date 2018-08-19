import cntk as C
from cntk.layers import *
import numpy as np
import h5py
from helpers import HighwayBlock
class _ElmoCharEncoder(object):
    def __init__(self,weight_file='elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'):
        self.weight_file= weight_file
        self.filter_num = 7
        self.highway_num = 2

    def _load_weight(self):
        self._load_char_embed()
        self._load_cnn_weight()
        self._load_highway()
        self._load_proj()
    def _load_char_embed(self):
        with h5py.File(self.weight_file, 'r') as f:
            tmp_weight = f['char_embed'][...] # shape: 261*16
        weight = np.zeros((tmp_weight.shape[0]+1, tmp_weight.shape[1]), dtype=np.float32)
        weight[1:,:] = tmp_weight
        self.char_embed = C.constant(weight, name='elmo_char_embed')
    def _load_cnn_weight(self):
        self.convs = [None]*self.filter_num
        with h5py.File(self.weight_file,'r') as fin:
            for i in range(self.filter_num):
                weight = fin['CNN']['W_cnn_{}'.format(i)][...] # (1,h, w, out_c)
                bias = fin['CNN']['b_cnn_{}'.format(i)][...] # (int,)
                w_reshape = np.transpose(weight.squeeze(axis=0), axes=(2,0,1)) # (out_c, h, w) 
                b_reshape = np.reshape(bias,(len(bias),1,1))
                self.convs[i] = Convolution2D((w_reshape.shape[1] ,w_reshape.shape[2]), w_reshape.shape[0],
                    init=w_reshape, reduction_rank=0, activation=C.relu,
                    init_bias=b_reshape, name='char_conv_{}'.format(i))
    def _load_highway(self):
        self.highways = [None]*self.highway_num
        with h5py.File(self.weight_file,'r') as fin:
            for i in range(self.highway_num):
                w_transform = fin['CNN_high_{}'.format(i)]['W_transform'][...] # use for C.times(x,W)
                b_transform = fin['CNN_high_{}'.format(i)]['b_transform'][...]
                w_carry = fin['CNN_high_{}'.format(i)]['W_carry'][...] # use for (1-g)x+g*f(x)
                b_carry = fin['CNN_high_{}'.format(i)]['b_carry'][...]
                self.highways[i] = HighwayBlock(w_transform.shape[0],
                    transform_weight_initializer=w_transform,
                    transform_bias_initializer=b_transform,
                    update_weight_initializer=w_carry,
                    update_bias_initializer=b_carry)
    def _load_proj(self):
        with h5py.File(self.weight_file,'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]
            W_proj = C.constant(weight)
            b_proj = C.constant(bias)
        @C.Function
        def dense(x):
            return C.relu(C.times(x, W_proj)+b_proj)
        self.proj = dense
    def build(self, require_train=False):
        self._load_weight()
        @C.Function
        def _func(x):
            input_ph = C.placeholder()

            ph = C.placeholder()
            onehot_value = C.one_hot(ph,262)
            x1 = C.times(onehot_value, self.char_embed) # [#,*][50,16]
            # x2 = self.convs[0](x1) # [#,*][32,50,1]
            convs_res = []
            for i in range(self.filter_num):
                conv_res = self.convs[i](x1)
                convs_res.append(C.reshape(C.reduce_max(conv_res, axis=1),(-1,)))
            token_embed = C.splice(*convs_res) # [#,*][2048]
            
            tmp_res = token_embed
            for i in range(self.highway_num):
                tmp_res = self.highways[i](tmp_res)
            highway_out=tmp_res # [#,*][2048]
            proj_out = self.proj(highway_out) # [#,*][512]

            if not require_train:
                res = proj_out.clone(C.CloneMethod.freeze, {ph:input_ph})
            else:
                res = proj_out.clone(C.CloneMethod.clone, {ph:input_ph})
            return C.as_block(
                res,[(input_ph, x)], 'elmo_char_encoder', 'elmo_char_encoder'
            )
        return _func
    def test(self):
        input_ph=C.sequence.input_variable((50,))
        encoder = self.build()
        encode_out = encoder(input_ph)
        return encode_out
        
    def lstm(dh, dc, x):

        dhs = Sdh(dh)  # previous values, stabilized
        dcs = Sdc(dc)
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        proj4 = b + times(x, W) + times(dhs, H)

        it_proj  = slice (proj4, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        bit_proj = slice (proj4, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ft_proj  = slice (proj4, stack_axis, 2*stacked_dim, 3*stacked_dim)
        ot_proj  = slice (proj4, stack_axis, 3*stacked_dim, 4*stacked_dim)

        # helper to inject peephole connection if requested
        def peep(x, c, C):
            return x + C * c if use_peepholes else x

        it = sigmoid (peep (it_proj, dcs, Ci))        # input gate(t)
        # TODO: should both activations be replaced?
        bit = it * activation (bit_proj)              # applied to tanh of input network

        ft = sigmoid (peep (ft_proj, dcs, Cf))        # forget-me-not gate(t)
        bft = ft * dc                                 # applied to cell(t-1)

        ct = bft + bit                                # c(t) is sum of both

        ot = sigmoid (peep (ot_proj, Sct(ct), Co))    # output gate(t)
        ht = ot * activation (ct)                     # applied to tanh(cell(t))

        c = ct                                        # cell value
        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        # returns the new state as a tuple with names but order matters
        #return (Function.NamedOutput(h=h), Function.NamedOutput(c=c))
        return (h, c)

def proj_LSTM(input_dim, out_dim, init_W, init_H, init_b, init_W_0):
    '''numpy initial'''
    W = C.Constant(shape=(input_dim, 4096*4),value=init_W) # (512,4096*4)
    H = C.Constant(shape=(out_dim, 4*4096), value=init_H)
    b = C.Constant(shape=(4096*4,), value=init_b)
    proj_W = C.Constant(shape=(4096,out_dim), value=init_W_0)
    stacked_dim=4096
    @C.Function
    def unit(dh, dc, x):
        ''' dh: out_dim, dc:4096, x:input_dim'''
        proj4 = b + times(x, W) + times(dh, H)
        it_proj  = proj4[0:1*stacked_dim]  # split along stack_axis
        bit_proj = proj4[1*stacked_dim: 2*stacked_dim]
        ft_proj  = proj4[2*stacked_dim: 3*stacked_dim]
        ot_proj  = proj4[3*stacked_dim: 4*stacked_dim]

        it = C.sigmoid(it_proj)        # input gate(t)
        # TODO: should both activations be replaced?
        bit = it * C.tanh(bit_proj)              # applied to tanh of input network

        ft = C.sigmoid (ft_proj)        # forget-me-not gate(t)
        bft = ft * dc                                 # applied to cell(t-1)

        ct = bft + bit                                # c(t) is sum of both

        ot = C.sigmoid (ot_proj)    # output gate(t)
        ht = ot * C.tanh(ct)                     # applied to tanh(cell(t))

        c = ct                                        # cell value
        h = ht
        proj_h = C.times(h, proj_W) # out_dim
        return (proj_h, c) 
    return unit

class _ElmoBilm(object):
    def __init__(self, weight_file='elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'):
        self.weight_file = weight_file
        self.layer_num = 2
        self.forward_unit = [None for _ in range(self.layer_num)]
        self.backward_unit = [None for _ in range(self.layer_num)]
        self.dropout=0.2

    def _load_weight(self):
        with h5py.File(self.weight_file,'r') as fin:
            for i_layer in range(self.layer_num): 
                for j_direction in range(2): # forward and backward
                    dataset = fin['RNN_%s' % j_direction]['RNN']['MultiRNNCell']['Cell%s' % i_layer]['LSTMCell']
                    # tensorflow packs the gates as input, memory, forget, output as same as cntk
                    tf_weights = dataset['W_0'][...] # (1024, 16384)
                    init_W = tf_weights[:512,:]
                    init_H = tf_weights[512:,:]
                    tf_bias = dataset['B'][...] # (16384,)
                    # tensorflow adds 1.0 to forget gate bias instead of modifying the
                    # parameters...
                    tf_bias[4096*2:4096*3] += 1.0
                    proj_weights = dataset['W_P_0'][...] # (4096, 512)
                    
                    if j_direction==0:
                        self.forward_unit[i_layer] = proj_LSTM(512,512,init_W, init_H, tf_bias, proj_weights)
                    else:
                        self.backward_unit[i_layer] = proj_LSTM(512,512,init_W, init_H, tf_bias, proj_weights)
    def build(self):
        self._load_weight()
        layer1_f = Recurrence(self.forward_unit[0])
        layer1_b = Recurrence(self.backward_unit[0],True)
        layer2_f = Recurrence(self.forward_unit[1])
        layer2_b = Recurrence(self.backward_unit[1], True)
        drop = Dropout(self.dropout)
        @C.Function
        def _func(x):
            layer1_out_f = drop(layer1_f(x)); layer1_out_b = drop(layer1_b(x))
            layer2_out_f = drop(layer2_f(layer1_out_f))
            layer2_out_b = drop(layer2_b(layer1_out_b))
            return (
                C.splice(layer1_out_f,layer1_out_b), C.splice(layer2_out_f,layer2_out_b)
                )
        return _func
    def test(self):
        input_ph = C.sequence.input_variable(512)
        model = self.build()
        return model(input_ph)
class ElmoEmbedder(object):
    def __init__(self, weight_file='elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'):
        self.dropout = 0.2
        self.encoder_fac = _ElmoCharEncoder(weight_file)
        self.bilm_fac = _ElmoBilm(weight_file)
    def build(self, require_train=False):
        gamma = C.Parameter(1,init=1)
        scales = C.Parameter(3, init=C.glorot_uniform(), name='scales')
        encoder = self.encoder_fac.build()
        bilm = self.bilm_fac.build()
        @C.Function
        def _func(x):
            ph = C.placeholder()
            first_out = encoder(ph)
            second_out, third_out = bilm(first_out).outputs # [#,*][1024]
            dup_first_out = C.splice(first_out, first_out) #[#,*][1024]
            s = C.softmax(scales)
            out = gamma*(s[0]*dup_first_out+s[1]*second_out+s[2]*third_out)
            return C.as_block(
                out, [(ph, x)],'Elmo', 'Elmo'
            )
        return _func


if __name__=="__main__":
    model = ElmoEmbedder().build()
    words = C.sequence.input_variable(50)
    res = model(words)

    a = np.random.randint(261,size=(1,10,50))
    my_array = res.eval({words:a})
