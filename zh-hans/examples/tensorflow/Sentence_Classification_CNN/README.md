# Windows上的Sentence_Classification_CNN

[简体中文](/zh-hans/examples/tensorflow/Sentence_Classification_CNN/README.md)

This is a reimplementation of [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)(EMNLP 14')

# 说明

1. data_process.py用于解析xml并枚举其中内容。
2. Sentence_Classification_CNN.py是算法的卷积模型。
3. train.py定义了可以调整的模型参数。

# 数据

Training.xml is the trainning set of NLPCC2013 task2. Test.xml is the test set of NLPCC2013 task2.

# 许可证

The code is under MIT license