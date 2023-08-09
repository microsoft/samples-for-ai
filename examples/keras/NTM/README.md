### Changelog 0.2:
* API CHANGE: Controller models now must have linear activation. The activation of the NTM-Layer is selected
  by the new parameter "activation" (default: "linear"). For all the stuff that interacts with the memory we now
  have very precise handselected activations which asume that there was no prior de-linearisation.
  This requirement on the controller will probably be final.
* There is now support for multiple read/write heads! Use the parameters read_heads resp. write_heads at initialisation
  (by default both are 1).
* The code around controller output splitting and activation was completely rewritten and cleaned from a lot of
  copy-paste-code.
* Unfortunately we lost backend neutrality: As tf.slice is used extensivly, we have to either try getting K.slice or
  have to do a case distinction over backend. Use the old version if you need another backend than tensorflow! And
  please write me a message.
* As less activations have to be computed, it is now a tiny little bit faster (~1%).
* Stateful models do not work anymore. Actually they never worked, the testing routine was just broken. Will be repaired
  asap.

# The Neural Turing Machine
### Introduction
This code tries to implement the Neural Turing Machine, as found in 
https://arxiv.org/abs/1410.5401, as a backend neutral recurrent keras layer.

A very default experiment, the copy task, is provided, too.

In the end there is a TODO-List. Help would be appreciated!



### User guide
For a quick start on the copy task, type 

    python main.py -v ntm

while in a python enviroment which has tensorflow, keras and numpy.
Having tensorflow-gpu is recommend, as everything is about 20x faster.
In my case this experiment takes about 100 minutes on a NVIDIA GTX 1050 Ti.
The -v is optional and offers much more detailed information about the achieved accuracy, and also after every training
epoch.
Logging data is written LOGDIR_BASE, which is ./logs/ by default. View them with tensorboard:

    tensorboard --logdir ./logs

If you've luck and not had a terrible run (that can happen, unfortunately), you now have a machine capable of copying a
given sequence! I wonder if we could have achieved that any other way ...

These results are especially interesting compared to an LSTM model: Run

    python main.py lstm

This builds 3 layers of LSTM with and goes through the same testing procedure
as above, which for me resulted in a training time of approximately 1h (same GPU) and 
(roughly) 100%, 100%, 94%, 50%, 50% accuracy at the respective test lengths.
This shows that the NTM has advantages over LSTM in some cases. Especially considering the LSTM model has about 807.200
trainable parameters while the NTM had a mere 3100! 

Have fun playing around, maybe with other controllers? dense, double_dense and lstm are build in.


### API
From the outside, this implementation looks like a regular recurrent layer in keras.
It has however a number of non-obvious parameters:

#### Hyperparameters

  
*  `n_width`: This is the width of the memory matrix. Increasing this increases computational complexity in O(n^2). The
   controller shape is not dependant on this, making weight transfer possible.

*  `m_depth`: This is the depth of the memory matrix. Increasing this increases the number of trainable weights in O(m^2). It also changes controller shape. 

*  `controller_model`: This parameter allows you to place a keras model of appropriate shape as the controller. The
appropriate shape can be calculated via controller_input_output_shape. If None is set, a single dense layer will be
used. 

*  `read_heads`: The number of read heads this NTM should have. Has quadratic influence on the number of trainable
   weights. Default: 1

*  `write_heads`: The number of write heads this NTM should have. Has quadratic influence on the number of trainable
   weights, but for small numbers a *huge* impact. Default: 1


#### Usage

More or less minimal code example:

    from keras.models import Sequential
    from keras.optimizers import Adam
    from ntm import NeuralTuringMachine as NTM

    model = Sequential()
    model.name = "NTM_-_" + controller_model.name

    ntm = NTM(output_dim, n_slots=50, m_depth=20, shift_range=3,
              controller_model=None,
              return_sequences=True,
              input_shape=(None, input_dim), 
              batch_size = 100)
    model.add(ntm)

    sgd = Adam(lr=learning_rate, clipnorm=clipnorm)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                   metrics = ['binary_accuracy'], sample_weight_mode="temporal")

What if we instead want a more complex controller? Design it, e.g. double LSTM:

    controller = Sequential()
    controller.name=ntm_controller_architecture
    controller.add(LSTM(units=150,
                        stateful=True,
                        implementation=2,   # best for gpu. other ones also might not work.
                        batch_input_shape=(batch_size, None, controller_input_dim)))
    controller.add(LSTM(units=controller_output_dim,
                        activation='linear',
                        stateful=True,
                        implementation=2))   # best for gpu. other ones also might not work.

    controller.compile(loss='binary_crossentropy', optimizer=sgd,
                     metrics = ['binary_accuracy'], sample_weight_mode="temporal")

And now use the same code as above, only with controller_model=controller.

Note that we used linear as the last activation layer! This is of critical importance.
The activation of the NTM-layer can be set the parameter activation (default: linear).

Note that a correct controller_input_dim and controller_output_dim can be calculated via controller_input_output_shape:

    from ntm import controller_input_output_shape
    controller_input_dim, controller_output_dim = ntm.controller_input_output_shape(
                input_dim, output_dim, m_depth, n_slots, shift_range, read_heads, write_heads) 


Also note that every statefull controller must carry around his own state, as was done here with 

    stateful=True





## TODO:
- [x] Arbitrary number of read and write heads
- [ ] Support of masking, and maybe dropout, one has to reason about it theoretically first.
- [ ] Support for get and set config to better enable model saving
- [x] A bit of code cleaning: especially the controller output splitting is ugly as hell.
- [x] Support for arbitrary activation functions would be nice, currently restricted to sigmoid.
- [ ] Make it backend neutral again! Some testing might be nice, too. 
- [ ] Maybe add the other experiments of the original paper?
- [ ] Mooaaar speeeed. Look if there are platant performance optimizations possible. 
