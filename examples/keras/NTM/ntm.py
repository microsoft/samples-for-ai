

import warnings
import numpy as np
import tensorflow as tf

import keras
from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras.layers.core import Dense
from keras.initializers import RandomNormal, Orthogonal, Zeros, Constant
from keras import backend as K
from keras.engine.topology import InputSpec 
from keras.activations import get as get_activations
from keras.activations import softmax, tanh, sigmoid, hard_sigmoid

def _circulant(leng, n_shifts):
 
    """

    Paramters:
    ----------
    leng: int > 0, number of memory locations
    n_shifts: int > 0, number of allowed shifts (if 1, no shift)

    Returns:
    --------
    shift operation, a tensor with dimensions (n_shifts, leng, leng)
    """
    eye = np.eye(leng)
    shifts = range(n_shifts//2, -n_shifts//2, -1)
    C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    return K.variable(C.astype(K.floatx()))


def _renorm(x):
    return x / (K.sum(x, axis=1, keepdims=True))


def _cosine_distance(M, k):
    # this is equation (6), or as I like to call it: The NaN factory.
    # TODO: Find it in a library (keras cosine loss?)
    # normalizing first as it is better conditioned.
    nk = K.l2_normalize(k, axis=-1)
    nM = K.l2_normalize(M, axis=-1)
    cosine_distance = K.batch_dot(nM, nk)
    # TODO: Do succesfull error handling
    #cosine_distance_error_handling = tf.Print(cosine_distance, [cosine_distance], message="NaN occured in _cosine_distance")
    #cosine_distance_error_handling = K.ones(cosine_distance_error_handling.shape)
    #cosine_distance = tf.case({K.any(tf.is_nan(cosine_distance)) : (lambda: cosine_distance_error_handling)},
    #        default = lambda: cosine_distance, strict=True)
    return cosine_distance

def _controller_read_head_emitting_dim(m_depth, shift_range):
    # For calculating the controller output dimension, we need the output_dim of the whole layer
    # (which is only passed during building) plus all the stuff we need to interact with the memory,
    # calculated here:
    #
    # For every read head the addressing data (for details, see figure 2):
    #       key_vector (m_depth) 
    #       beta (1)
    #       g (1)
    #       shift_vector (shift_range)
    #       gamma (1)
    return (m_depth + 1 + 1 + shift_range + 1)

def _controller_write_head_emitting_dim(m_depth, shift_range):
    controller_read_head_emitting_dim = _controller_read_head_emitting_dim(m_depth, shift_range)
    # But what do for write heads? The adressing_data_dim is the same, but we emit additionally:
    #       erase_vector (m_depth)
    #       add_vector (m_depth)
    return controller_read_head_emitting_dim + 2*m_depth

def controller_input_output_shape(input_dim, output_dim, m_depth, n_slots, shift_range, read_heads, write_heads):
    controller_read_head_emitting_dim = _controller_read_head_emitting_dim(m_depth, shift_range)
    controller_write_head_emitting_dim = _controller_write_head_emitting_dim(m_depth, shift_range)

    # The controller output size consists of 
    #       the regular output dim
    # plus, for every read and write head the respective dims times the number of heads.
    controller_output_dim = (output_dim + 
            read_heads * controller_read_head_emitting_dim + 
            write_heads * controller_write_head_emitting_dim)
    # For the input shape of the controller the formula is a bit easier:
    #       the regular input_dim 
    # plus, for every read head:
    #       read_vector (m_depth).
    # So that results in:
    controller_input_dim = input_dim + read_heads * m_depth

    return controller_input_dim, controller_output_dim


class NeuralTuringMachine(Recurrent):
    """ Neural Turing Machines

    Non obvious parameter:
    ----------------------
    shift_range: int, number of available shifts, ex. if 3, avilable shifts are
                 (-1, 0, 1)
    n_slots: Memory width, defined in 3.1 as N
    m_depth: Memory depth at each location, defined in 3.1 as M
    controller_model: A keras model with required restrictions to be used as a controller.
                        The requirements are appropriate shape, linear activation and stateful=True if recurrent.
                        Default: One dense layer.
    activation: This is the activation applied to the layer output.
                        It can be either a Keras activation or a string like "tanh", "sigmoid", "linear" etc.
                        Default is linear.

    Known issues:
    -------------
    Currently batch_input_size is necessary. Or not? Im not even sure :(

    """
    def __init__(self, units, 
                        n_slots=50,
                        m_depth=20,
                        shift_range=3,
                        controller_model=None,
                        read_heads=1,
                        write_heads=1,
                        activation='sigmoid',
                        batch_size=777,                 
                        stateful=False,
                        **kwargs):
        self.output_dim = units
        self.units = units
        self.n_slots = n_slots
        self.m_depth = m_depth
        self.shift_range = shift_range
        self.controller = controller_model
        self.activation = get_activations(activation)
        self.read_heads = read_heads
        self.write_heads = write_heads
        self.batch_size = batch_size

#        self.return_sequence = True
        try:
            if controller.state.stateful:
                self.controller_with_state = True 
        except:
            self.controller_with_state = False


        self.controller_read_head_emitting_dim = _controller_read_head_emitting_dim(m_depth, shift_range)
        self.controller_write_head_emitting_dim = _controller_write_head_emitting_dim(m_depth, shift_range)

        super(NeuralTuringMachine, self).__init__(**kwargs)

    def build(self, input_shape):
        bs, input_length, input_dim = input_shape

        self.controller_input_dim, self.controller_output_dim = controller_input_output_shape(
                input_dim, self.units, self.m_depth, self.n_slots, self.shift_range, self.read_heads,
                self.write_heads)
            
        # Now that we've calculated the shape of the controller, we have add it to the layer/model.
        if self.controller is None:
            self.controller = Dense(
                name = "controller",
                activation = 'linear',
                bias_initializer = 'zeros',
                units = self.controller_output_dim,
                input_shape = (bs, input_length, self.controller_input_dim))
            self.controller.build(input_shape=(self.batch_size, input_length, self.controller_input_dim))
            self.controller_with_state = False


        # This is a fixed shift matrix
        self.C = _circulant(self.n_slots, self.shift_range)

        self.trainable_weights = self.controller.trainable_weights 

        # We need to declare the number of states we want to carry around.
        # In our case the dimension seems to be 6 (LSTM) or 5 (GRU) or 4 (FF),
        # see self.get_initial_states, those respond to:
        # [old_ntm_output] + [init_M, init_wr, init_ww] +  [init_h] (LSMT and GRU) + [(init_c] (LSTM only))
        # old_ntm_output does not make sense in our world, but is required by the definition of the step function we
        # intend to use.
        # WARNING: What self.state_spec does is only poorly understood,
        # I only copied it from keras/recurrent.py.
        self.states = [None, None, None, None]
        self.state_spec = [InputSpec(shape=(None, self.output_dim)),                            # old_ntm_output
                            InputSpec(shape=(None, self.n_slots, self.m_depth)),                # Memory
                            InputSpec(shape=(None, self.read_heads, self.n_slots)),   # weights_read
                            InputSpec(shape=(None, self.write_heads, self.n_slots))]  # weights_write

        super(NeuralTuringMachine, self).build(input_shape)


    def get_initial_state(self, X):
        #if not self.stateful:
        #    self.controller.reset_states()

        init_old_ntm_output = K.ones((self.batch_size, self.output_dim), name="init_old_ntm_output")*0.42 
        init_M = K.ones((self.batch_size, self.n_slots , self.m_depth), name='main_memory')*0.042
        init_wr = np.zeros((self.batch_size, self.read_heads, self.n_slots))
        init_wr[:,:,0] = 1
        init_wr = K.variable(init_wr, name="init_weights_read")
        init_ww = np.zeros((self.batch_size, self.write_heads, self.n_slots))
        init_ww[:,:,0] = 1
        init_ww = K.variable(init_ww, name="init_weights_write")
        return [init_old_ntm_output, init_M, init_wr, init_ww]




    # See chapter 3.1
    def _read_from_memory(self, weights, M):
        # see equation (2)
        return K.sum((weights[:, :, None]*M),axis=1)

    # See chapter 3.2
    def _write_to_memory_erase(self, M, w, e):
        # see equation (3)
        M_tilda = M * (1 - w[:, :, None]*e[:, None, :])
        return M_tilda

    def _write_to_memory_add(self, M_tilda, w, a):
        # see equation (4)
        M_out = M_tilda + w[:, :, None]*a[:, None, :]
        return M_out

    # This is the chain described in Figure 2, or in further detail by
    # Chapter 3.3.1 (content based) and Chapter 3.3.2 (location based)
    # C is our convolution function precomputed above.
    def _get_weight_vector(self, M, w_tm1, k, beta, g, s, gamma):
#        M = tf.Print(M, [M, w_tm1, k], message='get weights beg1: ')
#        M = tf.Print(M, [beta, g, s, gamma], message='get weights beg2: ')
        # Content adressing, see Chapter 3.3.1:
        num = beta * _cosine_distance(M, k)
        w_c  = K.softmax(num) # It turns out that equation (5) is just softmax.
        # Location adressing, see Chapter 3.3.2:
        # Equation 7:
        w_g = (g * w_c) + (1-g)*w_tm1
        # C_s is the circular convolution
        #C_w = K.sum((self.C[None, :, :, :] * w_g[:, None, None, :]),axis=3)
        # Equation 8:
        # TODO: Explain
        C_s = K.sum(K.repeat_elements(self.C[None, :, :, :], self.batch_size, axis=0) * s[:,:,None,None], axis=1)
        w_tilda = K.batch_dot(C_s, w_g)
        # Equation 9:
        w_out = _renorm(w_tilda ** gamma)

        return w_out

    def _run_controller(self, inputs, read_vector):
        controller_input = K.concatenate([inputs, read_vector])
        if self.controller_with_state or len(self.controller.input_shape) == 3:
            controller_input = controller_input[:,None,:]
            controller_output = self.controller.call(controller_input)
            if self.controller.output_shape == 3:
                controller_output = controller_output[:,0,:]
        else:
            controller_output = self.controller.call(controller_input)

        return controller_output

    def _split_and_apply_activations(self, controller_output):
        """ This takes the controller output, splits it in ntm_output, read and wright adressing data.
            It returns a triple of ntm_output, controller_instructions_read, controller_instructions_write.
            ntm_output is a tensor, controller_instructions_read and controller_instructions_write are lists containing
            the adressing instruction (k, beta, g, shift, gamma) and in case of write also the writing constructions,
            consisting of an erase and an add vector. 

            As it is necesseary for stable results,
            k and add_vector is activated via tanh, erase_vector via sigmoid (this is critical!),
            shift via softmax,
            gamma is sigmoided, inversed and clipped (probably not ideal)
            g is sigmoided,
            beta is linear (probably not ideal!) """
        
        # splitting
        ntm_output, controller_instructions_read, controller_instructions_write = tf.split(
                    controller_output,
                    np.asarray([self.output_dim,
                                self.read_heads * self.controller_read_head_emitting_dim,
                                self.write_heads * self.controller_write_head_emitting_dim]),
                    axis=1)

        controller_instructions_read = tf.split(controller_instructions_read, self.read_heads, axis=1)
        controller_instructions_write = tf.split(controller_instructions_write, self.write_heads, axis=1)

        controller_instructions_read = [
                tf.split(single_head_data, np.asarray([self.m_depth, 1, 1, 3, 1]), axis=1) for 
                single_head_data in controller_instructions_read]
        
        controller_instructions_write = [
                tf.split(single_head_data, np.asarray([self.m_depth, 1, 1, 3, 1, self.m_depth, self.m_depth]), axis=1) for 
                single_head_data in controller_instructions_write]
        
        #activation
        ntm_output = self.activation(ntm_output)
        controller_instructions_read = [(tanh(k), hard_sigmoid(beta)+0.5, sigmoid(g), softmax(shift), 1 + 9*sigmoid(gamma)) for
                (k, beta, g, shift, gamma) in controller_instructions_read]
        controller_instructions_write = [
                (tanh(k), hard_sigmoid(beta)+0.5, sigmoid(g), softmax(shift), 1 + 9*sigmoid(gamma), hard_sigmoid(erase_vector), tanh(add_vector))  for 
                (k, beta, g, shift, gamma, erase_vector, add_vector) in controller_instructions_write]
       
        return (ntm_output, controller_instructions_read, controller_instructions_write)
        


    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim


    def step(self, layer_input, states):
        # As a step function MUST return its regular output as the first element in the list of states,
        # we have _ here.
        _, M, weights_read_tm1, weights_write_tm1 = states[:4]

        # reshaping (TODO: figure out how save n-dimensional state) 
        weights_read_tm1 = K.reshape(weights_read_tm1,
                (self.batch_size, self.read_heads, self.n_slots))
        weights_write_tm1 = K.reshape(weights_write_tm1, 
                (self.batch_size, self.write_heads, self.n_slots))

        # We have the old memory M, and a read weighting w_read_tm1 calculated in the last
        # step. This is enough to calculate the read_vector we feed into the controller:
        memory_read_input = K.concatenate([self._read_from_memory(weights_read_tm1[:,i], M) for 
                i in range(self.read_heads)])

        # Now feed the controller and let it run a single step, implemented by calling the step function directly,
        # which we have to provide with the actual input from outside, the information we've read an the states which
        # are relevant to the controller.
        controller_output = self._run_controller(layer_input, memory_read_input)

        # We take the big chunk of unactivated controller output and subdivide it into actual output, reading and
        # writing instructions. Also specific activions for each parameter are applied.
        ntm_output, controller_instructions_read, controller_instructions_write = \
                self._split_and_apply_activations(controller_output)


        # Now we want to write to the memory for each head. We have to be carefull about concurrency, otherwise there is
        # a chance the write heads will interact with each other in unintended ways!
        # We first calculate all the weights, then perform all the erasing and only after that the adding is done.
        # addressing:
        weights_write = []
        for i in range(self.write_heads):
            write_head = controller_instructions_write[i]
            old_weight_vector = weights_write_tm1[:,i]
            weight_vector = self._get_weight_vector(M, old_weight_vector, *tuple(write_head[:5]))
            weights_write.append(weight_vector)
        # erasing:
        for i in range(self.write_heads):
            M = self._write_to_memory_erase(M, weights_write[i], controller_instructions_write[i][5])
        # adding:
        for i in range(self.write_heads):
            M = self._write_to_memory_add(M, weights_write[i], controller_instructions_write[i][6])

        # Only one thing left until this step is complete: Calculate the read weights we save in the state and use next
        # round:
        # As reading is side-effect-free, we dont have to worry about concurrency.
        weights_read = []
        for i in range(self.read_heads):
            read_head = controller_instructions_read[i]
            old_weight_vector = weights_read_tm1[:,i]
            weight_vector = self._get_weight_vector(M, old_weight_vector, *read_head)
            weights_read.append(weight_vector)

        # M = tf.Print(M, [K.mean(M), K.max(M), K.min(M)], message="Memory overview")
        # Now lets pack up the state in a list and call it a day.
        return ntm_output, [ntm_output, M, K.stack(weights_read, axis=1), K.stack(weights_write, axis=1)] 

