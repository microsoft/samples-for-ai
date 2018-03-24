# Content
This is a CNTK implement of [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831)

## Effectiveness
In InnRNN.py, a model trying to fit addition result of sequences over 2000 timestep are trained. When use IndRNN cell, 
the training loss continueously decrease and quickly convergence.

But the training loss nearly maintains its level when a LSTM cell is used.