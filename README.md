# Samples for AI

[![MIT licensed](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/Microsoft/samples-for-ai/blob/master/LICENSE)
[![Pull requests](https://img.shields.io/github/issues-pr-raw/Microsoft/samples-for-ai.svg)](https://github.com/Microsoft/samples-for-ai/pulls?q=is%3Aopen+is%3Apr)
[![Issues](https://img.shields.io/github/issues-raw/Microsoft/samples-for-ai.svg)](https://github.com/Microsoft/samples-for-ai/issues?q=is%3Aopen+is%3Aissue)

Samples for AI is a deep learning samples and projects collection. It contains a lot of classic deep learning algorithms and applications with different frameworks, which is a good entry for the beginners to get started with deep learning.

Samples in Visual Studio solution format are provided for users to get started with deep learning using:
- [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai)
- [Open Platform for AI](https://github.com/Microsoft/pai)
- Command line

Each solution has one or more sample projects.
Solutions are separated by different deep learning frameworks they use:
- CNTK (both BrainScript and Python languages)
- TensorFlow
- PyTorch
- Caffe2
- Keras
- MXNet
- Chainer
- Theano


# Getting Started

### 1. [Prerequisites](./docs/prerequisites.md)
   **Using a one-click installer to setup deep learning frameworks** has been moved to [**here**](./docs/prerequisites.md#Using_a_one-click_installer_to_setup_deep_learning_frameworks), please visit it for details.

### 2. [Download Data](./docs/download_data.md)

### 3. Run Samples

   - [Local Run](./docs/local_run.md)

   - [Submit Job to OpenPAI](./docs/submit_job_to_pai.md)


# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Related Projects

[Open Platform for AI](https://github.com/Microsoft/pai): an open source platform that provides complete AI model training and resource management capabilities, it is easy to extend and supports on-premise, cloud and hybrid environments in various scale.

[NeuronBlocks](https://github.com/Microsoft/NeuronBlocks) : A NLP deep learning modeling toolkit that helps engineers to build DNN models like playing Lego. The main goal of this toolkit is to minimize developing cost for NLP deep neural network model building, including both training and inference stages.

# License

Most of the samples scripts are from official github of each framework. They are under different licenses.


The scripts of CNTK are under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

The scripts of Tensorflow samples are under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0).
There are no changes to the original code.

For the scripts of Caffe2, different versions released with different licenses.
Currently, the master branch is under Apache 2.0 license. But the version 0.7 and 0.8.1 were released with [BSD 2-Clause license](https://github.com/caffe2/caffe2/tree/v0.8.1).
The scripts in our solution are based on caffe2 GitHub source tree version 0.7 and 0.8.1, with BSD 2-Clause license.

The scripts of Keras are under [MIT license](https://github.com/fchollet/keras/blob/master/LICENSE).

The scripts of Theano are under [BSD license](https://en.wikipedia.org/wiki/BSD_licenses).

The scripts of MXNet are under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0).
There are no changes to the original code.

The scripts of Chainer are under [MIT license](https://github.com/chainer/chainer/blob/master/LICENSE).
