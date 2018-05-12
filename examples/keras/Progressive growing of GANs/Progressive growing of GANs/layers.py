import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
from keras import activations
import tensorflow as tf
import numpy as np


#----------------------------------------------------------------------------
# Resize activation tensor 'inputs' of shape 'si' to match shape 'so'.
# 
class ACTVResizeLayer(Layer):
    def __init__(self,si,so,**kwargs):
        self.si = si
        self.so = so
        super(ACTVResizeLayer,self).__init__(**kwargs)
    def call(self, v, **kwargs):
        assert len(self.si) == len(self.so) and self.si[0] == self.so[0]

        # Decrease feature maps.  Attention: channels last
        if self.si[-1] > self.so[-1]:
            v = v[...,:self.so[-1]]

        # Increase feature maps.  Attention:channels last
        if self.si[-1] < self.so[-1]:
            z = K.zeros((self.so[:-1] + (self.so[-1] - self.si[-1])),dtype=v.dtype)
            v = K.concatenate([v,z])
        
        # Shrink spatial axis
        if len(self.si) == 4 and (self.si[1] > self.so[1] or self.si[2] > self.so[2]):
            assert self.si[1] % self.so[1] == 0 and self.si[2] % self.so[2] == 0
            pool_size = (self.si[1] / self.so[1],self.si[2] / self.so[2])
            strides = pool_size
            v = K.pool2d(v,pool_size=pool_size,strides=strides,padding='same',data_format='channels_last',pool_mode='avg')

        #Extend spatial axis
        for i in range(1,len(self.si) - 1):
            if self.si[i] < self.so[i]:
                assert self.so[i] % self.si[i] == 0
                v = K.repeat_elements(v,rep=int(self.so[i] / self.si[i]),axis=i)

        return v
    def compute_output_shape(self, input_shape):
        return self.so


#----------------------------------------------------------------------------
# Resolution selector for fading in new layers during progressive growing.
class LODSelectLayer(Layer):
    def __init__(self,cur_lod,first_incoming_lod=0,ref_idx=0, min_lod=None, max_lod=None,**kwargs):
        super(LODSelectLayer,self).__init__(**kwargs)
        self.cur_lod = cur_lod
        self.first_incoming_lod = first_incoming_lod
        self.ref_idx = ref_idx
        self.min_lod = min_lod
        self.max_lod = max_lod

    def call(self, inputs):
        self.input_shapes = [K.int_shape(input) for input in inputs]
        v = [ACTVResizeLayer(K.int_shape(input), self.input_shapes[self.ref_idx])(input) for input in inputs]
        lo = np.clip(int(np.floor(self.min_lod - self.first_incoming_lod)), 0, len(v)-1) if self.min_lod is not None else 0
        hi = np.clip(int(np.ceil(self.max_lod - self.first_incoming_lod)), lo, len(v)-1) if self.max_lod is not None else len(v)-1
        t = self.cur_lod - self.first_incoming_lod
        r = v[hi]
        for i in range(hi-1, lo-1, -1): # i = hi-1, hi-2, ..., lo
            r = K.switch(K.less(t, i+1), v[i] * ((i+1)-t) + v[i+1] * (t-i), r)
        if lo < hi:
            r = K.switch(K.less_equal(t, lo), v[lo], r)
        return r
    def compute_output_shape(self, input_shape):
        return self.input_shapes[self.ref_idx]



#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
class PixelNormLayer(Layer):
    def __init__(self,**kwargs):
        super(PixelNormLayer,self).__init__(**kwargs)
    def call(self, inputs, **kwargs):
        return inputs / K.sqrt(K.mean(inputs**2, axis=-1, keepdims=True) + 1.0e-8)
    def compute_output_shape(self, input_shape):
        return input_shape


# Copyright (c) 2016 Tim Salimans
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from the original implementation by Tim Salimans.
# Source: https://github.com/ceobillionaire/improved_gan/blob/master/mnist_svhn_cifar10/nn.py

# We modified it to Keras canonical form
class MinibatchLayer(Layer):
    def __init__(self,num_kernels,dim_per_kernel = 5,theta = None,log_weight_scale = None,b = None,init = False,**kwargs):
        super(MinibatchLayer,self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.dim_per_kernel = dim_per_kernel
        self.theta_arg = theta
        self.log_weight_scale_arg =log_weight_scale
        self.b_arg = b
        self.init_arg = init
    def build(self,input_shape):
        num_inputs = int(np.prod(input_shape[1:]))
        self.theta = self.add_weight(name = 'theta',shape =  (num_inputs, self.num_kernels, self.dim_per_kernel),initializer='zeros')
        if self.theta_arg == None:
            K.set_value(self.theta,K.random_normal((num_inputs, self.num_kernels, self.dim_per_kernel),0.0,0.05))
        self.log_weight_scale = self.add_weight(name ='log_weight_scale', shape= (self.num_kernels, self.dim_per_kernel),initializer='zeros')
        if self.log_weight_scale_arg == None:
            K.set_value(self.log_weight_scale,K.constant(0.0,shape = (self.num_kernels, self.dim_per_kernel)))
        self.kernel = self.theta * K.expand_dims(K.permute_dimensions((K.exp(self.log_weight_scale)/K.sqrt(K.sum(K.square(self.theta),axis=0))),[0,1]),0)
        self.bias = self.add_weight(name = 'bias',shape = (self.num_kernels,),initializer='zeros')
        if self.b_arg == None:
            K.set_value(self.bias,K.constant(-1.0,shape = (self.num_kernels,)))
        super(MinibatchLayer, self).build(input_shape)
    def call(self,input,**kargs):
        if K.ndim(input) > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = K.flatten(input)
        actv = K.batch_dot(input,self.kernel,[[1],[0]])
        abs_dif = (K.sum(K.abs(K.expand_dims(K.permute_dimensions(actv,[0,1,2]))-K.expand_dims(K.permute_dimensions(actv,[1,2,0]),0)),axis = 2)+
                   1e6*K.expand_dims(K.eye(K.int_shape(input)[0]),1))
        if self.init_arg:
            mean_min_abs_dif = 0.5 * K.mean(K.min(abs_dif, axis=2),axis=0)
            abs_dif/=K.expand_dims(K.expand_dims(mean_min_abs_dif,0))
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-K.expand_dims(K.log(mean_min_abs_dif)))]
        f = K.sum(K.exp(-abs_dif),axis = 2)

        if self.init_arg:
            mf = K.mean(f,axis=0)
            f -= K.expand_dims(mf,0)
            self.init_updates += [(self.bias,-mf)]
        else:
            f += K.expand_dims(self.bias,0)

        return K.concatenate([input,f],axis = 1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)





#----------------------------------------------------------------------------
# Applies equalized learning rate to the preceding layer.

class WScaleLayer(Layer):
    def __init__(self,incoming,activation = None,**kwargs):
        self.incoming = incoming
        self.activation = activations.get(activation)
        super(WScaleLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        kernel = K.get_value(self.incoming.kernel)
        scale = np.sqrt(np.mean(kernel ** 2))
        K.set_value(self.incoming.kernel,kernel/scale)
        self.scale=self.add_weight(name = 'scale',shape = scale.shape,trainable=False,initializer='zeros')
        K.set_value(self.scale,scale)
        super(WScaleLayer, self).build(input_shape)
        #if  hasattr(self.incoming, 'bias') and self.incoming.bias is not None:
        #    bias = K.get_value(self.incoming.bias)
        #    self.bias=self.add_weight(name = 'bias',shape = bias.shape,initializer='zeros')
            # del self.incoming.trainable_weights[self.incoming.bias]
            # self.incoming.bias = None
        
    def call(self, input, **kwargs):
        input = input * self.scale
        #if self.bias is not None:
        #    pattern = ['x'] + ['x'] * (K.ndim(input) - 2)+[0]
        #    input = input + K.expand_dims(K.expand_dims(K.expand_dims(self.bias,0),0),0)
        return self.activation(input)
    def compute_output_shape(self, input_shape):
        return input_shape


#----------------------------------------------------------------------------
# Applies bias

class AddBiasLayer(Layer):
    def __init__(self,**kwargs):
        super(AddBiasLayer,self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias', 
                                      shape=(input_shape[-1],),
                                      initializer='zeros',
                                      trainable=True)
        super(AddBiasLayer, self).build(input_shape)
        
    def call(self, input, **kwargs):
        if self.bias is not None:
            input = K.bias_add(input,self.bias)
        return input
    def compute_output_shape(self, input_shape):
        return input_shape



#----------------------------------------------------------------------------
# Minibatch stat concatenation layer.
# - func is the function to use for the activations across minibatch
# - averaging tells how much averaging to use ('all', 'spatial', 'none')
class MinibatchStatConcatLayer(Layer):
    def __init__(self,averaging = 'all',**kwargs):
        self.averaging = averaging.lower()
        super(MinibatchStatConcatLayer,self).__init__(**kwargs)
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: K.sqrt(K.mean((x - K.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)
    def call(self, input, **kwargs):
        s = list(K.int_shape(input))
        s[0] = tf.shape(input)[0]
        vals = self.adjusted_std(input,axis=0,keepdims=True)                # per activation, over minibatch dim
        if self.averaging == 'all':                                 # average everything --> 1 value per minibatch
            vals = K.mean(vals,keepdims=True)
            reps = s; reps[-1]=1;reps[0] = tf.shape(input)[0]
            vals = K.tile(vals,reps)
        elif self.averaging == 'spatial':                           # average spatial locations
            if len(s) == 4:
                vals = K.mean(vals,axis=(1,2),keepdims=True)
            reps = s; reps[-1]=1
            vals = K.tile(vals,reps)
        elif self.averaging == 'none':                              # no averaging, pass on all information
            vals = K.repeat_elements(vals,rep=s[0],axis=0)
        elif self.averaging == 'gpool':                             # EXPERIMENTAL: compute variance (func) over minibatch AND spatial locations.
            if len(s) == 4:
                vals = self.adjusted_std(input,axis=(0,1,2),keepdims=True)
            reps = s; reps[-1]=1
            vals = K.tile(vals,reps)
        elif self.averaging == 'flat':
            vals = self.adjusted_std(input,keepdims=True)                   # variance of ALL activations --> 1 value per minibatch
            reps = s; reps[-1]=1
            vals = K.tile(vals,reps)
        elif self.averaging.startswith('group'):                    # average everything over n groups of feature maps --> n values per minibatch
            n = int(self.averaging[len('group'):])
            vals = vals.reshape((1, s[1], s[2], n,s[3]/n))
            vals = K.mean(vals, axis=(1,2,4), keepdims=True)
            vals = vals.reshape((1, 1, 1,n))
            reps = s; reps[-1] = 1
            vals = K.tile(vals, reps)
        else:
            raise ValueError('Invalid averaging mode', self.averaging)
        return K.concatenate([input, vals], axis=-1)
    def compute_output_shape(self, input_shape):
        s = list(input_shape)
        if self.averaging == 'all': s[-1] += 1
        elif self.averaging == 'flat': s[-1] += 1
        elif self.averaging.startswith('group'): s[-1] += int(self.averaging[len('group'):])
        else: s[-1] *= 2
        return tuple(s)


#----------------------------------------------------------------------------
# Generalized dropout layer.  Supports arbitrary subsets of axes and different
# modes.  Mainly used to inject multiplicative Gaussian noise in the network.
class GDropLayer(Layer):
    def __init__(self,mode='mul', strength=0.4, axes=(0,3), normalize=False,**kwargs):
        super(GDropLayer,self).__init__(**kwargs)
        assert mode in ('drop', 'mul', 'prop')
        #self.random     = K.random_uniform(1, minval=1, maxval=2147462579, dtype=tf.float32, seed=None, name=None)
        self.mode       = mode
        self.strength   = strength
        self.axes       = [axes] if isinstance(axes, int) else list(axes)
        self.normalize  = normalize # If true, retain overall signal variance.
        self.gain       = None      # For experimentation.
    def call(self, input,deterministic=False, **kwargs):
        if self.gain is not None:
            input = input * self.gain
        if deterministic or not self.strength:
            return input

        in_shape  = self.input_shape
        in_axes   = range(len(in_shape))
        in_shape  = [in_shape[axis] if in_shape[axis] is not None else input.shape[axis] for axis in in_axes] # None => Theano expr
        rnd_shape = [in_shape[axis] for axis in self.axes]
        broadcast = [self.axes.index(axis) if axis in self.axes else 'x' for axis in in_axes]
        one       = K.constant(1)

        if self.mode == 'drop':
            p = one - self.strength
            rnd = K.random_binomial(tuple(rnd_shape), p=p, dtype=input.dtype) / p

        elif self.mode == 'mul':
            rnd = (one + self.strength) ** K.random_normal(tuple(rnd_shape), dtype=input.dtype)

        elif self.mode == 'prop':
            coef = self.strength * K.constant(np.sqrt(np.float32(self.input_shape[1])))
            rnd = K.random_normal(tuple(rnd_shape), dtype=input.dtype) * coef + one

        else:
            raise ValueError('Invalid GDropLayer mode', self.mode)

        if self.normalize:
            rnd = rnd / K.sqrt(K.mean(rnd ** 2, axis=3, keepdims=True))
        return input * K.permute_dimensions(rnd,broadcast)
    def compute_output_shape(self, input_shape):
        return input_shape


#----------------------------------------------------------------------------
# Layer normalization.  Custom reimplementation based on the paper:
# https://arxiv.org/abs/1607.06450
class LayerNormLayer(Layer):
    def __init__(self,incoming,epsilon,**kwargs):
        super(LayerNormLayer,self).__init__(**kwargs)
        self.incoming = incoming
        self.epsilon = epsilon
    def build(self,input_shape):
        gain = np.float32(1.0)
        self.gain = self.add_weight(name='gain',shape = gain.shape,  trainable=True,initializer='zeros')
        K.set_value(self.gain,gain)
        self.bias = None
        if hasattr(self.incoming, 'bias') and self.incoming.bias is not None: # steal bias
            bias = K.get_value(self.incoming.bias)
            self.bias = self.add_param(name = 'bias',shape = bias.shape)
            K.set_value(self.bias,bias)
            # del self.incoming.params[self.incoming.bias]
            # self.incoming.bias = None
        self.activation = activations.get('linear')
        if hasattr(self.incoming, 'activation') and self.incoming.activation is not None: # steal nonlinearity
            self.activation = self.incoming.activation
            self.incoming.activation = activations.get('linear')

        super(LayerNormLayer, self).build(input_shape)
    def call(self, input, **kwargs):
        avg_axes = range(1, len(self.input_shape()))
        input = input - K.mean(input, axis=avg_axes, keepdims=True) # subtract mean
        input = input * 1.0/K.sqrt(K.mean(K.square(input), axis=avg_axes, keepdims=True) + self.epsilon) # divide by stdev
        input = input * self.gain # multiply by gain
        if self.bias is not None:
            pattern = ['x'] + ['x'] * (K.ndim(input) - 2)+[0]
            input = input + K.expand_dims(K.expand_dims(K.expand_dims(self.bias,0),0),0)
        return self.activation(input)
    def compute_output_shape(self, input_shape):
        return input_shape

class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output-inputs[i]
        return output

class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        # Create a trainable weight variable for this layer.
        super(GradNorm, self).build(input_shapes)  # Be sure to call this somewhere!

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)