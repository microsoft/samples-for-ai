# 内容

[English](/examples/cntk/python/NumpyExperiment/README.md)

这是[Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN（独立循环神经网络：构建更长更深的RNN）](https://arxiv.org/abs/1803.04831)的CNTK实现。

## 效率

在InnRNN.py里，经过2000步时长的训练后，模型会被用来拟合结果。 在使用IndRNN单元时，训练损失会非常快的持续收敛。

但训练损失值会维持在与LSTM单元接近的水平。

MIT许可 http://www.opensource.org/licenses/mit-license.php