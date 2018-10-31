from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from skimage.viewer import ImageViewer, CollectionViewer

m = Sequential()
m.add(Dense(100, input_dim=30))

def weights_viewer(path):
    w = []
    for i in range(1000):
        filename = "model.ckpt.{:04d}.hdf5".format(i)
        m.load_weights(path + filename)
        k, b = m.get_weights()
        i = np.concatenate((k, np.ones((2,100)), b[None,]))
        w.append(i)
    CollectionViewer(w).show()

weights_viewer("logs/2017-08-19_17:54:43_-_NTM_-_dense/")
