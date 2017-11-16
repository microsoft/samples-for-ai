# Introduction 

Samples in Visual Studio solution format are provided for users to get started with deep learning using [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai).
Each solution has one or more sample projects.
Solutions are separated by different deep learning frameworks they use:
- CNTK (both BrainScript and Python languages)
- Tensorflow
- Caffe2
- Keras
- MXNet
- Chainer
- Theano

# Getting Started

## Prerequisites to run the samples
- Install [Microsoft Visual Studio](https://www.visualstudio.com/) 2017 or 2015.
- Install [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai).
- Pre-download data
    - For CNTK BrainScript MNIST project, in the "input" folder, run "python install_mnist.py" to download data.

## Runing samples locally

- CNTK BrainScript Projects
    - Set the project you want to run as "Startup Project".
    - Set the script you want to run as "Startup File".
    - Click "Run CNTK Brain Script".

- Python Projects
    - Set the "Startup File".
    - Right click the startup Python script, and click "Start without Debugging" or "Start with Debugging" context menus.


# License

The samples scripts are from official github of each framework. They are under different licenses.

The scripts of CNTK are under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

The scripts of Tensorflow samples are under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0).
There are no changes on the original code.

For the scripts of Caffe2, different versions released with different licenses. 
Currently, the master branch is under Apache 2.0 license. But the version 0.7 and 0.8.1 were released with [BSD 2-Clause license](https://github.com/caffe2/caffe2/tree/v0.8.1).
The scripts in our solution are based on caffe2 github source tree version 0.7 and 0.8.1, with BSD 2-Clause license.

The scripts of Keras are under [MIT license](https://github.com/fchollet/keras/blob/master/LICENSE).

The scripts of Theano are under [BSD license](https://en.wikipedia.org/wiki/BSD_licenses).

The scripts of MXNet are under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0).
There are no changes on the original code.

The scripts of Chainer are under [MIT license](https://github.com/chainer/chainer/blob/master/LICENSE).

