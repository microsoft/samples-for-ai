# 用基于采样的软最大值(SoftMax) 方法来构建神经网络语言模型

[English](/examples/cntk/python/ptb/README.md)

此样例演示了如何用采样的软最大值方法来训练基于令牌的神经网络语言模型。 此模型在给定一些词语的情况下，使用软最大值来计算下一个词的概率，并进行预测。 由于词库的数量非常大，因此最后的软最大值步骤会非常的花时间。

基于采样的软最大值是用来减少训练时间的方法。 详情查看[基于采样的软最大值教程](https://github.com/Microsoft/CNTK/blob/v2.0.beta12.0/Tutorials/CNTK_207_Training_with_Sampled_Softmax.ipynb)

注意，这里提供的数据集只有1万个不同的词语。 这个数量不算多，因此基于采样的软最大值方法还不能展示出对性能上的显著提升。 显著的性能收益需要更大的词汇表才能更明显。

## 指南

此样例使用的Penn Treebank数据没有保存在GitHub中，需要先下载。 请运行一次download_data.py来下载数据。 这条命令会创建一个./ptb的文件夹，这里面包含了样例所需的所有数据。

运行word-rnn.py来训练模型。 word-rnn的主要章节定义了一些参数来控制训练过程。

* `use_sampled_softmax`用来在基于采样的软最大值和全软最大值之间切换。
* `softmax_sample_size`设置在基于采样的软最大值中随机样本的数量。