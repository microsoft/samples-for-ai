import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam

lr = 4e-4
clipnorm = 10
units = 256



def gen_model(input_dim=10, output_dim=8, batch_size=100):
    model_dense = Sequential()
    model_dense.name = "FFW"
    model_dense.batch_size = batch_size
    model_dense.input_dim = input_dim
    model_dense.output_dim = output_dim

    model_dense.add(Dense(input_shape=(None, input_dim), units=output_dim))
    model_dense.add(Activation('sigmoid'))

    sgd = Adam(lr=lr, clipnorm=clipnorm)
    model_dense.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['binary_accuracy'], sample_weight_mode="temporal")

    return model_dense
