# Samples for AI

[![MIT licensed](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/Microsoft/samples-for-ai/blob/master/LICENSE)

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


# Getting Started

## Prerequisites to run the samples
- Install [Microsoft Visual Studio](https://www.visualstudio.com/) 2017 or 2015.
- Install [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai).
- Pre-download data
    - For CNTK BrainScript MNIST project, in the `input` folder, run `python install_mnist.py` to download data.

## Preparing development environment
Before training deep learning models on your local or remote computer, please make sure you have the deep learning software installed.
This includes the latest drivers and libraries for your NVIDIA GPU (if you have one). You also need to install Python and libraries such as NumPy, SciPy, Python support for Visual Studio, and frameworks such as Microsoft Cognitive Toolkit (CNTK), TensorFlow, Caffe2, MXNet, Keras, Theano, PyTorch and/or Chainer.

Please visit [here](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md) for detailed instruction.

## Using a one-click installer to setup deep learning frameworks

Currently, this installer works on Windows, macOS and Linux:

- Install latest NVIDIA GPU driver, CUDA 9.0, and cuDNN 7.0 if applicable.
- Install latest **Python 3.5 or 3.6**. Other Python versions are not supported.
- Run the following commands in a terminal:
    > ### NOTE
    >
    > - If your Python distribution is installed in the system directory (e.g. the one shipped with Visual Studio 2017, or the built-in one on Linux), administrative permission (e.g. "sudo" on Linux) is required to launch the installer.
    > - Pass "**--user**" argument, if you want to install to the Python user install directory for your platform. Typically `~/.local/`, or `%APPDATA%\Python` on Windows.
    > - The installer will detect whether NVIDIA GPU cards are available and set up software for CUDA 9.0 by default. You can pass "**--cuda80**" argument to force installing software for CUDA 8.0 .
- Windows
    ```bash
    git clone https://github.com/Microsoft/samples-for-ai.git
    cd samples-for-ai
    cd installer
    python.exe install.py
    ```
- Linux and macOS
    ```bash
    git clone https://github.com/Microsoft/samples-for-ai.git
    cd samples-for-ai
    cd installer
    python3 install.py
    ```

## Running samples locally

- CNTK BrainScript Projects
    - Set the project you want to run as "Startup Project".
    - Set the script you want to run as "Startup File".
    - Click "Run CNTK Brain Script".

- Python Projects
    - Set the "Startup File".
    - Right click the startup Python script, and click "Start without Debugging" or "Start with Debugging" context menus.


## Submit samples to Microsoft PAI platform

[**Microsoft OpenPAI**](https://github.com/Microsoft/pai) is an open source platform that provides complete AI model training and resource management capabilities. These samples can submit to OpenPAI cluster. for each project, a JSON file is provided as an example to show how to configure the submit information. You can just use the example JSON file or set your specified configuration.

### How to submit a job to OpenPAI

- Right-Click project name -> "Submit Job...".
- In the pop-up dialog window, select your OpenPAI cluster.
- Write your own configuration or "Import" JSON file.
    - If you want to use example JSON file as configuration: Click "Import..." button, select one JSON file
- Click "Submit".

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
