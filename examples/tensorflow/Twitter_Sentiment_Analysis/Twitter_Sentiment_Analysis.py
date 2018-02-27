#MIT License
#if you have any question, contact me 
#2354558125@qq.com, Linfeng Zhang, Northeastern University In China
import tensorflow as tf



'''
lr: learning rate(int)
training_iters: number of training limit(int)
batch_size: size of each batch
n_inputs: the dimension of vectors after word embedding
n_steps: the time length of input
n_classes: the kinds of output
'''

def CNN_RNN(dataset,lr=0.001, training_iters = 1000, batch_size=100, n_inputs=28, n_steps=28, n_classes=10,):
    tf.set_random_seed(1)
    sess = tf.Session()
    n_hidden_unis = 128
    ''' functions and class used in the function(CNN_RNN) '''
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def RNN(X, weights, biases):

        # hidden layer for input to cell
        # X(128 batch, 28 steps, 28 inputs) => (128*28, 28)
        X = tf.reshape(X, [-1, n_inputs])
        # ==>(128 batch * 28 steps, 28 hidden)
        X_in = tf.matmul(X, weights['in']) + biases['in']
        # ==>(128 batch , 28 steps, 28 hidden)
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unis])
        # cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
        # lstm cell is divided into two parts(c_state, m_state)
        _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
        # hidden layer for output as the final results
        results = tf.matmul(states[1], weights['out']) + biases['out']  # states[1]->m_state states[1]=output[-1]
        # outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
        # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return tf.nn.tanh(results)


    x = tf.placeholder(tf.float32, [None, n_inputs, n_steps])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    w_conv1 = weight_variable([5, 5, 1, 32])
    ''' 5x5 is the size of patch(kernel), 1 is the input channel, 32 is the output channel '''
    b_conv1 = bias_variable([32])
    ''' each node has one bias '''
    ''' 2 layers of conv '''
    x_image = tf.reshape(x, [-1, n_inputs, n_steps, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    ''' full connected layer'''
    w_fc1 = weight_variable([n_inputs//4 * n_steps//4 * 64, 1024])
    b_fc1 = weight_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, n_inputs//4 * n_steps//4 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    ''' dropout layer '''
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024, n_classes])
    b_fc2 = bias_variable([n_classes])

    y_conv = tf.nn.tanh(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)



    weights = {
        # (28,128)
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
        # (128,10)
        'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
    }
    biases = {
        # (128,)
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
        # (10,)
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }


    pred = RNN(x, weights, biases)
    pred = (tf.add(pred, y_conv))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_prob: 1.0
            })
            step += 1
    return sess



