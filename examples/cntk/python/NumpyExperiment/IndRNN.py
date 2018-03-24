import cntk as C
from cntk.initializer import xavier, glorot_uniform, normal
from cntk.logging import ProgressPrinter
import numpy as np

class IndRNN(object):
    def __init__(self, hidden_dim,input_size,
        recurrent_min_abs=0,
        recurrent_max_abs=None,
        recurrent_kernel_initializer=1.0,
        input_kernel_initializer=normal(0.001),
        activation=C.relu,
        name=None):
        self._hidden_dim=hidden_dim
        self._recurrent_min_abs=recurrent_min_abs
        self._recurrent_max_abs=recurrent_max_abs
        self._recurrent_initializer=recurrent_kernel_initializer
        self._input_initializer=input_kernel_initializer
        self._activation=activation
        self._input_size=input_size

    def build(self):
        input_kernel = C.Parameter(shape=(self._input_size, self._hidden_dim), init=self._input_initializer)
        recur_kernel = C.Parameter(shape=(self._hidden_dim,), init=self._recurrent_initializer)
        bias = C.Parameter(shape=(self._hidden_dim), init=0)
        
        if self._recurrent_min_abs>0:
            abs_kernel = C.abs(recur_kernel)
            min_abs_kernel = C.element_max(abs_kernel, self._recurrent_min_abs)
            recur_kernel = min_abs_kernel * C.element_select(C.greater_equal(recur_kernel,C.constant(0)), C.constant(1), C.constant(-1))

        if self._recurrent_max_abs:
            recur_kernel = C.clip(recur_kernel,-self._recurrent_max_abs, self._recurrent_max_abs)
        
        @C.Function
        def runit(h, x):
            h_t = C.times(x, input_kernel) + bias + recur_kernel*h
            return h_t
        return runit

def get_batch():
    inp = np.random.rand(20,2000,2)
    target = np.sum(inp,axis=(1,2))
    target = np.reshape(target, (20,1))
    return inp,target

if __name__=='__main__':
    HIDDEN_DIM=128

    input_ph=C.sequence.input_variable(2)
    targets_ph=C.input_variable(shape=1)
    
    model = C.layers.Sequential([
        C.layers.Recurrence(IndRNN(HIDDEN_DIM, 2).build()), 
        C.layers.Recurrence(IndRNN(1, HIDDEN_DIM).build()),
        C.sequence.last
        ])
    output = model(input_ph)

    loss = C.losses.squared_error(output, targets_ph)
    comp = C.combine(output, loss)

    lrs = [(1,0.02),(300,0.002),(600,0.0001)]
    lr_schedule = C.learners.learning_parameter_schedule(lrs)
    
    learner = C.learners.adam(loss.parameters, lr_schedule, 0.9)
    trainer = C.Trainer(output, loss, learner,ProgressPrinter(5))

    input, target = get_batch()
    for _ in range(1000):
        trainer.train_minibatch({input_ph:input, targets_ph:target})
    
    res = output.eval({input_ph:input})
    print('predict:{}, target:{}'.format(res,target))
    
    # just use lstm
    model2 = C.layers.Sequential([
        C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM)),
        C.sequence.last,
        C.layers.Dense(1, activation=C.relu)
    ])
    output = model2(input_ph)

    loss = C.losses.squared_error(output, targets_ph)
    comp = C.combine(output, loss)

    learner = C.learners.adam(loss.parameters, lr_schedule, 0.9)
    trainer = C.Trainer(output, loss, learner,ProgressPrinter(5))

    input, target = get_batch()
    for _ in range(1000):
        trainer.train_minibatch({input_ph:input, targets_ph:target})
    
    res = output.eval({input_ph:input})
    print('predict:{}, target:{}'.format(res,target))
    
