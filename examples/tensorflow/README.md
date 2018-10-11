﻿# Introduction

[简体中文](/zh-hans/examples/tensorflow/README.md)

These examples are from tensorflow examples and tensorflow model zoo tutorials, without any change.

## List of Projects

1. **MNIST**: ([source of the script](https://github.com/tensorflow/models/tree/master/tutorials/image/mnist))

This is an example for training & evaluating the MNIST network using a feed dictionary. MNIST is a popular dataset for handwritten digits recognition. It contains 60000 examples as training set and 10000 examples as test set.

For more details, you can see the tensorflow githubs. [Tensorflow](https://github.com/tensorflow/tensorflow) and [Tensorflow Model Zoo](https://github.com/tensorflow/models).


2. **cifar10**: ([source of the script](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10))

CIFAR-10 is a common benchmark in machine learning for image recognition. (http://www.cs.toronto.edu/~kriz/cifar.html)

The code is a simple example to train and evaluate a convolutional neural network(CNN) on CPU.

For more details, you can see the tensorflow githubs. [Tensorflow](https://github.com/tensorflow/tensorflow) and [Tensorflow Model Zoo](https://github.com/tensorflow/models).


3. **embedding**: ([source of the script](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec))

This is an example for word embedding. Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers.

The test data is downloaded from http://mattmahoney.net/dc/text8.zip

The test data for the Large Text Compression Benchmark is the first 109 bytes of the English Wikipedia dump on Mar. 3, 2006. http://download.wikipedia.org/enwiki/20060303/enwiki-20060303-pages-articles.xml.bz2 (1.1 GB or 4.8 GB after decompressing with bzip2 - link no longer works). Results are also given for the first 108 bytes, which is also used for the Hutter Prize.

We filter the 1 GB test file enwik9 to produce a 715 MB file fil9, and compress this with 17 compressors. Furthermore, we produce the file text8 by truncating fil9 to 100 MB, and test this on 25 compressors, including the 17 tested on fil9.  text8 is the first 108 bytes of fil9. (From: http://mattmahoney.net/dc/textdata.html )

For more details, you can see the tensorflow githubs. [Tensorflow](https://github.com/tensorflow/tensorflow) and [Tensorflow Model Zoo](https://github.com/tensorflow/models).


4. **imagenet**: ([source  of the script](https://github.com/tensorflow/models/tree/master/tutorials/image/imagenet))

This is an example for simple image classification with Inception model.

Data is downloaded from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

See the tutorial and website for a detailed description of how to use this script to perform image recognition. https://tensorflow.org/tutorials/image_recognition/

For more details, you can see the tensorflow githubs. [Tensorflow](https://github.com/tensorflow/tensorflow) and [Tensorflow Model Zoo](https://github.com/tensorflow/models).


5. **MemN2N**

This is TensorFlow implementation of End-To-End Memory Networks for language modeling. It use a set of sample Penn Tree Bank(PTB) as Dataset.


6. **NTM**

This is Tensorflow implementation of Neural Turing Machine. This implementation uses an LSTM controller. NTM models with multiple read/write heads are supported.


# How to Run

1. Open the solution.(It will open with Visual Studio 2017 by default.)

2. Right click the project name to set the project want to run as "Startup Project"

3. Right click the script name to set the script want to run as "Startup File"

4. Right-click the script -> Start without Debugging


# Contributors

Some projects are contributed by University Students from Microsoft Student Club. Below is the detail information.

1. Project MemN2N, NTM

    - Contributors: Tian-You Gao, Xiang Li
    - University: Nanjing University of Aeronautics and Astronautics
