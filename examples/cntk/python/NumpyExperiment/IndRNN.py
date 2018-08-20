#
# Copyright (c) 2018 Wang XX
#
# MIT License
# http://www.opensource.org/licenses/mit-license.php
#
import cntk as C
from cntk.initializer import xavier, glorot_uniform, normal
from cntk.ops.functions import UserFunction
from cntk.logging import ProgressPrinter
import numpy as np
import argparse
print(C.device.all_devices())
try:
    C.device.try_set_default_device(C.device.gpu(0))
    C.use_default_device()
except:
    C.device.try_set_default_device(C.device.cpu())
    C.use_default_device()
class IndRNNUnit(object):
    def __init__(self, hidden_dim,input_size,
        recurrent_min_abs=None,
        recurrent_max_abs=None,
        recurrent_kernel_initializer=1.0,
        input_kernel_initializer=normal(0.01),
        activation=C.relu,
        name=None):

        self._hidden_dim=hidden_dim
        self._recurrent_min_abs=recurrent_min_abs
        self._recurrent_max_abs=recurrent_max_abs
        self._recurrent_initializer=recurrent_kernel_initializer
        self._input_initializer=input_kernel_initializer
        self._activation=activation
        self._input_size=input_size

    def checkbound(self):
        if self._recurrent_max_abs:
            self.recur_kernel.value = np.clip(self.recur_kernel.value, -self._recurrent_max_abs, self._recurrent_max_abs)

        if self._recurrent_min_abs:
            abs_kernel = np.clip(np.abs(self.recur_kernel.value), a_min=self._recurrent_min_abs, a_max=np.inf)
            self.recur_kernel.value = np.sign(self.recur_kernel.value) * abs_kernel

        # print('[DEBUG] abs kernel', self.recur_kernel.value)

    def build(self):
        self.input_kernel = C.Parameter(shape=(self._input_size, self._hidden_dim), init=self._input_initializer)
        self.recur_kernel = C.Parameter(shape=(self._hidden_dim,), init=self._recurrent_initializer)
        self.bias = C.Parameter(shape=(self._hidden_dim), init=0)
        @C.Function
        def runit(h,x):
            ht = self._activation(C.times(x, self.input_kernel) + h*self.recur_kernel+self.bias)
            return ht
        return runit
def get_batch(N, seq_len):
    X_num = np.random.uniform(low=0, high=1, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in range(N):
        # Default uniform distribution on position sampling
        positions = np.random.choice(seq_len, size=2, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return X, Y


timesteps = 1000
RECURRENT_MAX = pow(2, 1 / timesteps)
U_lowbound=pow(1.0/2.0, 1.0 / timesteps)

bs = 50
base_lr = 0.2 #0.0002*bs

if __name__=='__main__':
    HIDDEN_DIM=128

    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--lr', default=0.2, type=float)
    parser.add_argument('--bs', default=50, help='batch size', type=int)
    parser.add_argument('--time_step', default=1000, help='how long a sequence is', type=int)
    args=parser.parse_args()

    bs = args.bs
    base_lr = args.lr
    timesteps = args.time_step

    lrs = [(t, base_lr*(10**(-i))) for i,t in enumerate(range(1, 60000, 20000))]
    print(lrs)
    lr_schedule = C.learners.learning_parameter_schedule(lrs)

    if not args.lstm:
        input_ph=C.sequence.input_variable(2)
        targets_ph=C.input_variable(shape=1)

        runit1 = IndRNNUnit(HIDDEN_DIM, 2, recurrent_max_abs=RECURRENT_MAX, recurrent_min_abs=0)
        runit2 = IndRNNUnit(HIDDEN_DIM, HIDDEN_DIM, recurrent_max_abs=RECURRENT_MAX, recurrent_min_abs=U_lowbound)
        model = C.layers.Sequential([
            C.layers.Recurrence(runit1.build()),
            C.layers.Fold(runit2.build()),
            C.layers.Dense(1, init_bias=0.1, init=C.normal(0.001))
            ])
        output = model(input_ph)

        loss = C.reduce_mean(C.square(output-targets_ph)) #C.losses.squared_error(output, targets_ph)
        comp = C.combine(output, loss)
        tensorboard_writer = C.logging.TensorBoardProgressWriter(bs, log_dir='.',model=loss)
        learner = C.learners.adam(loss.parameters, lr_schedule, 0.9)
        trainer = C.Trainer(output, loss, learner,[ProgressPrinter(20), tensorboard_writer])

        for step in range(60000):
            input, target = get_batch(500, timesteps)
            runit1.checkbound()
            runit2.checkbound()
            trainer.train_minibatch({input_ph:input, targets_ph:target})
            if step % 200==0:
                trainer.summarize_training_progress()
                print('[training indrnn] lr:', learner.learning_rate())

        res = output.eval({input_ph:input})
        print('predict:{}\ntarget:{}'.format(res,target))
    else:
        # === just use lstm ===
        input_ph2 = C.sequence.input_variable(2)
        targets_ph2 = C.input_variable(shape=1)
        model2 = C.layers.Sequential([
            C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM)),
            C.sequence.last,
            C.layers.Dense(1, init_bias=0.1, init=C.normal(0.001))
        ])
        output2 = model2(input_ph2)

        loss2 = C.losses.squared_error(output2, targets_ph2)
        comp2 = C.combine(output2, loss2)
        tensorboard_writer2 = C.logging.TensorBoardProgressWriter(bs, log_dir='.',model=loss2)
        learner2 = C.learners.adam(loss2.parameters, lr_schedule, 0.9)
        trainer2 = C.Trainer(output2, loss2, learner2, [ProgressPrinter(20), tensorboard_writer2])

        for step in range(60000):
            input, target = get_batch(10, timesteps)
            trainer2.train_minibatch({input_ph2:input, targets_ph2:target})
            if step % 200 == 0:
                trainer2.summarize_training_progress()

        res = output2.eval({input_ph2:input})
        print('predict:{}\ntarget:{}'.format(res,target))

