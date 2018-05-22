import cntk as C
from cntk.initializer import xavier, glorot_uniform, normal

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