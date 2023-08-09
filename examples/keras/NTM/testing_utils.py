from copyTask import get_sample
from datetime import datetime
import numpy as np
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN

LOG_PATH_BASE="logs/"     #this is for tensorboard callbacks




def test_model(model, sequence_length=None, verboose=False):
    input_dim = model.input_dim
    output_dim = model.output_dim
    batch_size = model.batch_size

    I, V, sw = next(get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim,
                                        max_size=sequence_length, min_size=sequence_length))
    Y = np.asarray(model.predict(I, batch_size=batch_size))

    if not np.isnan(Y.sum()): #checks for a NaN anywhere
        Y = (Y > 0.5).astype('float64')
        x = V[:, -sequence_length:, :] == Y[:, -sequence_length:, :]
        acc = x.mean() * 100
        if verboose:
            print("the overall accuracy for sequence_length {0} was: {1}".format(sequence_length, x.mean()))
            print("per bit")
            print(x.mean(axis=(0,1)))
            print("per timeslot")
            print(x.mean(axis=(0,2)))
    else:
        ntm = model.layers[0]
        weights = ntm.get_weights()
        import pudb; pu.db
        acc = 0
    return acc


def train_model(model, epochs=10, min_size=5, max_size=20, callbacks=None, verboose=False):
    input_dim = model.input_dim
    output_dim = model.output_dim
    batch_size = model.batch_size

    sample_generator = get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim,
                                                max_size=max_size, min_size=min_size)
    if verboose:
        for j in range(epochs):
            model.fit_generator(sample_generator, steps_per_epoch=10, epochs=j+1, callbacks=callbacks, initial_epoch=j)
            print("currently at epoch {0}".format(j+1))
            for i in [5,10,20,40]:
                test_model(model, sequence_length=i, verboose=True)
    else:
        model.fit_generator(sample_generator, steps_per_epoch=10, epochs=epochs, callbacks=callbacks)

    print("done training")


def lengthy_test(model, testrange=[5,10,20,40,80], epochs=100, verboose=True):
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = LOG_PATH_BASE + ts + "_-_" + model.name 
    tensorboard = TensorBoard(log_dir=log_path,
                                write_graph=False, #This eats a lot of space. Enable with caution!
                                #histogram_freq = 1,
                                write_images=True,
                                batch_size = model.batch_size,
                                write_grads=True)
    model_saver =  ModelCheckpoint(log_path + "/model.ckpt.{epoch:04d}.hdf5", monitor='loss', period=1)
    callbacks = [tensorboard, TerminateOnNaN(), model_saver]

    for i in testrange:
        acc = test_model(model, sequence_length=i, verboose=verboose)
        print("the accuracy for length {0} was: {1}%".format(i,acc))

    train_model(model, epochs=epochs, callbacks=callbacks, verboose=verboose)

    for i in testrange:
        acc = test_model(model, sequence_length=i, verboose=verboose)
        print("the accuracy for length {0} was: {1}%".format(i,acc))
    return
