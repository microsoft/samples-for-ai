# Introduction

CNTK examples for python lauguage binding. In this Visual Studio solution, there are six projects.

## List of Projects

1. **ATIS:**

This is an example for training feed forward and LSTM networks for speech data.

This project preprocess ATIS data by converting words into word indexes, and labels into label IDs in order to use CNTKTextFormatReader. 
You can use any script/tool to preprocess your text data files. In this project, data is already preprocessed.

2. **CIFAR10:**

The example applies CNN on the CIFAR-10 dataset. It is a popular image classification example. The network contains four convolution layers and three dense layers. 
Max pooling is conducted for every two convolution layers. Dropout is applied after the first two dense layers. It adds data augmentation to training.


3. **CMUDict:**

This is an example for sequence-to-sequence modeling for grapheme-to-phoneme (aka letter-to-sound) conversion on the CMUDict dictionary.

4. **MNIST:**

This is an example for training a feedforward classification model for MNIST images.

5. **NumpyExperiment:**

This is a CNTK implement of [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831)

6. **ptb:**

This is an example for creating and train an RNN language model. The dataset used in this project is a subset of the Penn Tree Bank dataset.


# How to Run

1. Open the "CNTKPythonExamples.sln" solution.(It will open with Visual Studio 2017 by default.)

2. Right click the project name to set the project want to run as "Startup Project"

3. Right click the script name to set the script want to run as "Startup File"

4. Right-click the script -> Start without Debugging


# Contributors

Some projects are contributed by University Students from Microsoft Student Club.

1. Project NumpyExperiment

    - Contributors: XiaoXi Wang (and TuXiangXiaoBai Team)
    - University: Sun Yat-sen University