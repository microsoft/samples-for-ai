# Introduction 
There are Visual Studio sample solutions provided, with which you can start with using VisualStudioToolsForAI extension. Solutions are separated according to different deep learning frameworks they used. Each solution has one or more sample projects with the same deep learning framework. Currently, we support these frameworks: 
- CNTK
- Tensorflow
- Caffe2
- Keras
- MXNet
- Chainer
- theano

The solutions have two kinds of projects. One is BrainScript project for CNTK BrainScript language. The other is Python flavored project, which support the frameworks had python bindings.


# Getting Started

## Prerequisites to run the samples
1. Download and Install VisualStudioToolsForAI Extension
a. Microsoft.VisualStudioToolsForAI.VS2017.vsix for Visual Studio 2017
b. Microsoft.VisualStudioToolsForAI.VS2015.vsix for Visual Studio 2015
c. Visual Studio Code Tools for AI for Visual Studio Code

2. Pre-download data
In CNTK-BrainScript-MNIST project, in “input” folder, run “python install_mnist.py” to download data to current folder.

## Run samples in Local Environment

1. Run CNTK-BrainScript Projects
a. Download and Open the solution
b. Set the project you want as “Startup Project”
c. Set the script you want to run as “Startup File”
d. Click “Run CNTK Brain Script”

2. Run Python Flavored Projects
a. Download and Open the solution
b. Set the “Startup Project” and “Startup File”
c. Right-click the script -> Start without Debugging

# Examples Brief Introduction

## CNTKBarinScriptExamples
I. Project AN4
1. Description:
This is an example for training feed forward networks for speech data. You only need to run script “FeedForward.cntk” for FeedForward training.
2. Data:
The data is a modified version of AN4 dataset pre-processed and optimized for CNTK end-to-end testing. The data uses the format required by the HTKMLFReader.
The AN4 dataset is a part of CMU audio databases.

II. Project CMUDict
1. Description:
This is an example demonstrates the use of CNTK for grapheme-to-phoneme (letter-to-sound) conversion using a sequence-to-sequence model with attention, using the CMUDict dictionary.
2. Data:
The data is CMUDict dictionary.

III. Project MNIST
1. Description:
This is an example demonstrates usage of the NDL (Network Description Language) to define networks.
Three example scripts are provided for demonstration of three different NDL files:
a. One hidden layer network 
b. Convolutional network 
c. Convolutional network using batch normalization for the convolutional and fully connected layers.
Run script ”01_OneHidden.cntk” for one hidden layer network.
Run script “02_OneConv.cntk” for convolutional network (2 convolutional and 2 max pooling layers).
Run script “04_OneConvBN.cntk” for batch normalization for the convolutional and fully connected layers (Batch normalization training only implements on GPU).
2. Data:
The digit images in the MNIST set were originally selected and experimented with by Chris Burges and Corinna Cortes using bounding-box normalization and centering. Yann LeCun's version which is provided on this page uses centering by center of mass within in a larger window. (From http://yann.lecun.com/exdb/mnist/)

## CNTKPythonExamples
I. Project ATIS
1. Description:
This is an example for training feed forward and LSTM networks for speech data.
2. Data:
CNTK distribution contains a subset of ATIS Datasets:

Hemphill, Charles, et al. ATIS0 Complete LDC93S4A. Web Download. Philadelphia: Linguistic Data Consortium, 1993.
Garofolo, John, et al. ATIS2 LDC93S5. Web Download. Philadelphia: Linguistic Data Consortium, 1993.
Dahl, Deborah, et al. ATIS3 Test Data LDC95S26. Web Download. Philadelphia: Linguistic Data Consortium, 1995.
Dahl, Deborah, et al. ATIS3 Training Data LDC94S19. Web Download. Philadelphia: Linguistic Data Consortium, 1994. (From ‘CNTK/LICENSE.md’)

ATIS data in this example is preprocessed by converting words into word indexes, and labels into label IDs in order to use CNTKTextFormatReader. You can use any script/tool to preprocess your text data files. In this example, data is already preprocessed.

II. Project CIFAR10
1. Description:
The example applies CNN on the CIFAR-10 dataset. It is a popular image classification example. The network contains four convolution layers and three dense layers. Max pooling is conducted for every two convolution layers. Dropout is applied after the first two dense layers. It adds data augmentation to training.
2. Data:
The [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) is a popular dataset for image classification, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is a labeled subset of the 80 million tiny images dataset.
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

III. Project CMUDict
1. Description:
This is an example for sequence-to-sequence modeling for grapheme-to-phoneme (aka letter-to-sound) conversion on the CMUDict dictionary.
2. Data:
The data is CMUDict dictionary.

IV. Project MNIST
1. Description:
This is an example for training a feedforward classification model for MNIST images.
2. Data:
The digit images in the [MNIST set](http://yann.lecun.com/exdb/mnist/) were originally selected and experimented with by Chris Burges and Corinna Cortes using bounding-box normalization and centering. Yann LeCun's version which is provided on this page uses centering by center of mass within in a larger window.

V. Project ptb
1. Description: 
This is an example for creating and train an RNN language model.
2. Data:
Licensed under the MIT license.
This example uses Penn Treebank Data.

CNTK distribution contains a subset of the data of The Penn Treebank Project:
Marcus, Mitchell, Beatrice Santorini, and Mary Ann Marcinkiewicz. Treebank-2 LDC95T7. Web Download. Philadelphia: Linguistic Data Consortium, 1995. (From ‘CNTK/LICENSE.md’)

## TensorflowExamples
I. Project cifar10
1. Description:
The code is a simple example to train and evaluate a convolutional neural network(CNN) on CPU.
2. Data:
The [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) is a popular dataset for image classification, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is a labeled subset of the 80 million tiny images dataset.
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

II. Project embedding
1. Description:
This is an example for word embedding.
2. Data:
Licensed under GPL
Website: [http://mattmahoney.net/dc/](http://mattmahoney.net/dc/)
We filter the 1 GB test file enwik9 to produce a 715 MB file fil9, and compress this with 17 compressors. Furthermore, we produce the file text8 by truncating fil9 to 100 MB, and test this on 25 compressors, including the 17 tested on fil9.  text8 is the first 108 bytes of fil9. (From: http://mattmahoney.net/dc/textdata.html)

III. Project imagenet
1. Description:
This is an example for simple image classification with Inception.
Please see the tutorial and [website](https://tensorflow.org/tutorials/image_recognition/) for a detailed description of how
to use this script to perform image recognition.
2. Data:
[Download Data](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)

IV. Project MNIST
1. Description:
This is an example for training & evaluating the MNIST network using a feed dictionary.
2. Data:
The digit images in the MNIST set were originally selected and experimented with by Chris Burges and Corinna Cortes using bounding-box normalization and centering. Yann LeCun's version which is provided on this page uses centering by center of mass within in a larger window. (From http://yann.lecun.com/exdb/mnist/)

## Caffe2Examples
I. Project cifar10
1. Description:
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) is a common benchmark in machine learning for image recognition. The code is a simple example to train cifar10 with AlexNet model.
2. Data:
The CIFAR-10 dataset is a popular dataset for image classification, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is a labeled subset of the 80 million tiny images dataset.
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

II. Project MNIST
1. Description:
This is an example for training a LeNet model for MNIST images.
2. Data:
The digit images in the MNIST set were originally selected and experimented with by Chris Burges and Corinna Cortes using bounding-box normalization and centering. Yann LeCun's version which is provided on this page uses centering by center of mass within in a larger window. (From http://yann.lecun.com/exdb/mnist/)

## MXNetExamples
I. Project MNIST
1. Description:
Train mnist, see more explanation at [the website](http://mxnet.io/tutorials/python/mnist.html)
2. Data:
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)

## ChainerExamples
I. Project mnist
1. Description:
This is a minimal example to write a feed-forward net. The code consists of three parts: dataset preparation, network and optimizer definition and learning loop. This is a common routine to write a learning process of networks with dataset that is small enough to fit into memory.
2. Data:
MNIST dataset.

## KerasExamples
I. Project cifar10_tf
1. Description:
Train a simple deep CNN on the CIFAR10 small images dataset. With Tensorflow backend.
2. Data:
The [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) is a popular dataset for image classification, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is a labeled subset of the 80 million tiny images dataset.
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

II. Project cifar10_th
1. Description:
Train a simple deep CNN on the CIFAR10 small images dataset. With Theano backend.
2. Data:
CIFAR-10 data as above.

## TheanoExamples
I. Project LR
1. Description:
The examples is from [theano tutorial page](http://deeplearning.net/software/theano/tutorial/examples.html#a-real-example-logistic-regression). It is a simple logistic regression examples. And data is generated randomly.

# License

The samples scripts are from official github of each framework. They are under different license.

The script of CNTK is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

The script of Tensorflow samples are under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0). So we just move the original code here without any change.

The script of Caffe2(version 0.8.1) samples are under BSD 2-Clause license. (But currently, the code in master branch are under Apache 2.0 license)

The script of Keras is under [MIT license](https://github.com/fchollet/keras/blob/master/LICENSE).

The script of Theano is under [BSD license](https://en.wikipedia.org/wiki/BSD_licenses)

The script of MXNet is under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0). So we just move the original code here without any change.

The script of Chainer is under [MIT license](https://github.com/chainer/chainer/blob/master/LICENSE)

