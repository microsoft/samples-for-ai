import tensorflow as tf
import numpy as np

tiny = 1e-20

# Gumbel Softmax Trick
# >> implementation from Eric Jang
def sample_gumbel(shape, eps=1e-20):
    """sample from gumbel(0,1)"""
    U = tf.random_uniform(shape, minval=0,maxval=1)
    return -tf.log(-tf.log(U+eps)+eps)

def gumbel_softmax_sample(logits,temperature):
    """sample from gumbel-softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature )

def gumbel_softmax(logits, temperature = 1.0, hard = False):
    """
    Args:
        logits: [batch, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, directly take the argmax
    Returns:
        [batch_size, n_class] sample from Gumbel-Softmax
        when hard is True, return will be one-hot
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

#######
# Utils

def linear(x, out_dim, scope = None, reuse = None, bias = True,
           initializer = tf.contrib.layers.xavier_initializer()
          ):
    """
    Args:
        x: [batch, in_dim]
    """
    in_dim = x.get_shape()[-1]
    with tf.variable_scope(scope or 'linear', reuse = reuse):
        A = tf.get_variable('weights',[in_dim, out_dim], tf.float32,
                           initializer=initializer)
        if bias:
            b = tf.get_variable('bias', [out_dim], tf.float32,
                               initializer=tf.constant_initializer(0))
    if bias:
        return tf.nn.xw_plus_b(x, A, b)
    else:
        return tf.matmul(x, A)

def normal_KL(q_mu, q_log_var):
    return -0.5 * tf.reduce_sum(1 + q_log_var - tf.square(q_mu) - tf.exp(q_log_var), 1)

def categorical_KL(q, K):
    # q: prob distribution, output of softmax
    log_q = tf.log(q + 1e-20)
    KL = q*(log_q-tf.log(1.0/K))
    return tf.reduce_sum(KL, axis=1)

def entropy_loss(q):
    # q: prob distribution, [batch, k]
    return -tf.reduce_mean(tf.reduce_sum(q * tf.log(q + tiny), axis = 1))


def minimize_and_clip(optimizer, objective, var_list = None, clip_val=10, exclude = None):
    """
    Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            #gradients[i] = (tf.clip_by_value(grad, -clip_val, clip_val), var)
            if (exclude is None) or (var not in exclude):
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


############################
# Other NN Related
def mlp_net(x, dims, activation = tf.nn.elu,
            scope=None, reuse = None,
            dropout = None, is_training = None):
    if scope is None:
        scope = 'mlp-net'
    with tf.variable_scope(scope, reuse = reuse):
        for i,d in enumerate(dims):
            x = activation(linear(x,d,scope=scope+'-layer{i}'.format(i=i)))
            if dropout is not None and is_training is not None:
                x = tf.layers.dropout(x, rate=dropout, training=is_training)
    return x



################################################
# RNN Related
###################

def get_rnn_cell(dim, cell_name = 'lstm'):
    # stacked_rnn_cell
    if isinstance(cell_name,list) or isinstance(cell_name, tuple):
        if len(cell_name) == 1:
            return get_rnn_cell(dim, cell_name[0])
        cells = [get_rnn_cell(dim, c) for c in cell_name]
        return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    if cell_name == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
    elif cell_name == 'gru':
        return tf.contrib.rnn.GRUCell(dim)
    # other cell types not supported
    raise NotImplementedError

def get_rnn_state(x):
    # for stacked rnn cell, we return the state of last cell
    # for lstm cell, we return the hidden, h
    if isinstance(x, tuple):
        x = x[-1]
    if isinstance(x, tuple):
        # lstm cell
        x = x[-1]
    return x

def get_rnn_init_state(x, cell):
    """
    x: [batch, dim], must match the dim of the cell
    """
    if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
        batch = x.get_shape()[0]
        z = list(cell.zero_state(batch, dtype=tf.float32))
        if isinstance(z[0], tuple):
            z[0] = (tf.zeros_like(x), x)
        else:
            z[0] = x
        return tuple(z)
    if isinstance(cell.state_size, tuple):
        #lstm cell
        assert(len(cell.state_size) == 2)
        return (tf.zeros_like(x), x)
    # assume GRU Cell
    return x

def get_embedding(X, embedding, dropout = None, is_training = None):
    """
    Args:
        X:[batch, L]
        embedding: [N, dim]
        dropout: float, None when no dropout
        is_training: placeholder, whether training
    Returns:
        [batch, L, dim]
    """
    X_embed = tf.nn.embedding_lookup(embedding, X)
    if dropout is not None and is_training is not None:
        X_embed = tf.layers.dropout(X_embed, rate=dropout, training=is_training)
    return X_embed

#[TODO] to implement generate process and beam search
def myrnn(inputs, length, dim,
          global_input = None,
          init_val = None, cell_name = 'gru',
          scope = None, reuse = None):
    """
    Args:
        inputs: [batch, L, dim]
        global_input: [batch, _dim_], gloabl input fed to every input
        length: int, [batch]
        dim: dimension of hidden units
        init_val: initial state
        cell_name: string, either 'lstm' or 'gru'
    """
    with tf.variable_scope(scope or 'rnn', reuse = reuse):
        cell = get_rnn_cell(dim, cell_name)
        if init_val is not None:
            init_val = get_rnn_init_state(init_val, cell)
        if global_input is not None:
            L = inputs.get_shape()[1]
            exp_glob = tf.expand_dims(global_input,1)
            global_input = tf.tile(exp_glob,tf.stack([1,L,1]))
            inputs = tf.concat([inputs, global_input], axis=2)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs,
                                        sequence_length=length,
                                        initial_state = init_val,
                                        dtype=tf.float32,
                                        scope='dynamic-rnn')
        if isinstance(states, tuple):
            states = states[0]
    return outputs, states

def mybidrnn(inputs, length, dim,
             global_input = None,
             init_val = None, cell_name = 'gru',
             scope = None, reuse = None):
    """
    Args:
        inputs: [batch, L, dim]
        length: int, [batch]
        dim: dim of hidden units
        global_input: [batch, _dim_], global_input fed to every step
        init_val: initial state
        cell_name: string, either 'lstm' or 'gru'
    """
    with tf.variable_scope(scope or 'bidir-rnn', reuse = reuse):
        with tf.variable_scope('fwd-cell'):
            fw_cell = get_rnn_cell(dim, cell_name)
        with tf.variable_scope('bck-cell'):
            bw_cell = get_rnn_cell(dim, cell_name)
        if init_val is not None:
            # Assume fw_cell and bw_cell are of the same type
            init_val = get_rnn_init_state(init_vall, fw_cell)
        if global_input is not None:
            L = inputs.get_shape()[1]
            exp_glob = tf.expand_dims(global_input,1)
            global_input = tf.tile(exp_glob,tf.stack([1,L,1]))
            inputs = tf.concat([inputs, global_input], axis=2)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
                            fw_cell, bw_cell, inputs,
                            length, init_val, init_val,
                            dtype = tf.float32,
                            scope = 'bi-dynamic-rnn')
        fw_states, bw_states = states
        if isinstance(fw_states, tuple):
            fw_states = fw_states[0]
            bw_states = bw_states[0]
    return outputs, (fw_states, bw_states)

def sequence_softmax_loss(outputs, cat_n, labels, length, mask,
                          sampled_k = None, avg_batch = True, avg_length = False,
                          scope = None, var_scope = None, reuse = None):
    """
    Args:
        outputs: [batch, L, dim], outputs from a decoder RNN
        cat_n: int, number of categories
        labels: [batch, L], int label
        length: [batch], int, length of each batch
        mask: [batch, L], float, mask
        sampled_k: None when using a full softmax loss
        avg_batch: whether avg loss across the batch
        avg_length: whether avg loss over length
        return_vars: whether return the final project vars
    Returns:
        loss: a float number regarding loss
    """
    assert(len(outputs.get_shape()) == 3)
    _, L, dim = outputs.get_shape()
    with tf.variable_scope(var_scope or 'softmax_proj', reuse = reuse):
        # [NOTE] variable names compatible with linear()
        A_shape = [dim, cat_n] if sampled_k is None else [cat_n, dim]
        A = tf.get_variable('weights', A_shape, tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('bias', [cat_n], tf.float32,
                initializer=tf.constant_initializer(0))
    with tf.variable_scope(scope or 'sequence_softmax_loss'):
        if sampled_k is None:
            # full softmax loss
            logits = tf.nn.add_bias(tf.matmul(outputs, A), b)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                  logits=logits)
        else:
            # sampled softmax
            project_outputs = tf.reshape(outputs, tf.stack([-1, dim]))
            project_labels = tf.reshape(labels, [-1, 1])
            # [batch * L]
            loss = tf.nn.sampled_softmax_loss(A, b, project_labels, project_outputs, sampled_k, cat_n)
            loss = tf.reshape(loss, tf.stack([-1, L]))

        loss = tf.reduce_sum(loss * mask, axis=1)
        if avg_length:
            loss = loss / tf.cast(length,tf.float32)

        if avg_batch:
            ret_loss = tf.reduce_mean(loss)
        else:
            ret_loss = tf.reduce_sum(loss)
    return ret_loss

# RNN Greedy Decoder
#[TODO] to support multi-layer LSTM
def myrnn_decoder(inputs, length, dim, embedding, n_word,
                  global_input = None,
                  max_len = None,
                  init_val = None, cell_name = 'gru',
                  scope = None, var_scope = None, reuse = True):
    """
    Args:
        inputs: int, [batch, L]
        length: int, [batch], the length of the prefix in inputs
        global_input: [batch, dim], the input vector fed to every timestep
        dim: dim of hidden units
        init_val: initial_state
        sampling: sampling decoder or greedy decoder
    return:
        words: int, [batch, L]
    """
    if max_len is None:
        max_len = inputs.get_shape()[1]
    batch_seq_limit = tf.constant(max_len,dtype=tf.int32,
                                  shape=length.get_shape())#[batch]
    if init_val is not None:
        assert(init_val.get_shape()[1] == dim)
    with tf.variable_scope(var_scope or 'softmax_proj', reuse = reuse):
        # [NOTE] variable names compatible with linear()
        A = tf.get_variable('weights',[dim, cat_n], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('bias', [cat_n], tf.float32,
                initializer=tf.constant_initializer(0))
    # scope need to be compatible with rnn()
    with tf.variable_scope(scope or 'rnn', reuse = reuse):
        cell = get_rnn_cell(dim, cell_name)
        if init_val is not None:
            init_val = get_rnn_init_state(init_val, cell)

        ##########
        # define Loop Function
        def loop_fn(time, cell_output, cell_state, loop_state):
            next_loop_state = None # no loop state required
            if cell_output is None: # time 0
                emit_output = tf.constant(1, dtype=tf.int32)
                next_cell_state = init_val
            else: # normal timestep
                logits = tf.nn.add_bias(tf.matmul(cell_output, A), b)
                emit_output = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
                next_cell_state = cell_state
            elements_finished = (time >= batch_seq_limit)
            take_prefix = tf.cast(time < length, tf.int32)
            # int32, [batch]
            next_word = take_predix * (inputs[:, time] - emit_output) + emit_output
            next_input = tf.nn.embedding_lookup(embedding, next_word)
            if global_input is not None:
                next_input = tf.concat([next_input, global_input], axis = 1)
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)
        #########
        #[NOTE] scope must be compatible with myrnn()
        outputs_ta, _, _ = tf.nn.raw_rnn(cell,loop_fn,scope='dynamic-rnn')
        outputs = outputs_ta.stack()
    return outputs



############################
# Python Util
###########################
def clear_folder(folder):
    import os, shutil
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
