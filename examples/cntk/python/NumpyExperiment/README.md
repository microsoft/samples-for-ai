# Content

[简体中文](/zh-hans/examples/cntk/python/NumpyExperiment/README.md)

This is a CNTK implement of [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831)

## Effectiveness
In InnRNN.py, a toy example in whith two RNN models try to fit addition result of sequences over 2000 timestep.

The training loss nearly maintains its level when a LSTM cell is used,while
use IndRNN cell, the training loss continueously decrease and quickly convergence.

MIT License 
http://www.opensource.org/licenses/mit-license.php 