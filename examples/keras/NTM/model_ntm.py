import numpy as np

from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
import keras

from ntm import NeuralTuringMachine as NTM


n_slots = 128
m_depth = 20
learning_rate = 5e-4
clipnorm = 10

def gen_model(input_dim, batch_size, output_dim,
                n_slots=n_slots,
                m_depth=m_depth,
                controller_model=None,
                activation="sigmoid",
                read_heads = 1,
                write_heads = 1):

    model = Sequential()
    model.name = "NTM_-_" + controller_model.name
    model.batch_size = batch_size
    model.input_dim = input_dim
    model.output_dim = output_dim

    ntm = NTM(output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=3,
              controller_model=controller_model,
              activation=activation,
              read_heads = read_heads,
              write_heads = write_heads,
              return_sequences=True,
              input_shape=(None, input_dim), 
              batch_size = batch_size)
    model.add(ntm)

    sgd = Adam(lr=learning_rate, clipnorm=clipnorm)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['binary_accuracy'], sample_weight_mode="temporal")

    return model


