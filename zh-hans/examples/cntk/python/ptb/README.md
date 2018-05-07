# 用基于采样的软最大值(SoftMax) 方法来构建神经网络语言模型

[简体中文](/zh-hans/examples/cntk/python/ptb/README.md)

This example demonstrates how to use sampled softmax for training a token based neural language model. The model predicts the next word in a text given the previous ones where the probability of the next word is computed using a softmax. As the number of different words might be very high this final softmax step can turn out to be costly.

Sampled-softmax is a technique to reduce this cost at training time. For details see also the [sampled softmax tutorial](https://github.com/Microsoft/CNTK/blob/v2.0.beta12.0/Tutorials/CNTK_207_Training_with_Sampled_Softmax.ipynb)

Note the provided data set has only 10,000 distinct words. This number is still not very high and sampled softmax doesn't show any significant perf improvements here. The real perf gains will show up with larger vocabularies.

## 指南

This example uses Penn Treebank Data which is not stored in GitHub but must be downloaded first. To download the data please run download_data.py once. This will create a directory ./ptb that contains all the data we need for running the example.

Run word-rnn.py to train a model. The main section of word-rnn defines some parameters to control the training.

* `use_sampled_softmax`用来在基于采样的软最大值和全软最大值之间切换。
* `softmax_sample_size`设置在基于采样的软最大值中随机样本的数量。