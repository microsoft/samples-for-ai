# Prerequisites

## Install Tools for AI

- Install [Microsoft Visual Studio](https://www.visualstudio.com/) 2017 or 2015.

- Install [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai).

## Prepare development environment

Before training deep learning models on your local or remote computer, please make sure you have the deep learning software installed.
This includes the latest drivers and libraries for your NVIDIA GPU (if you have one). You also need to install Python and libraries such as NumPy, SciPy, Python support for Visual Studio, and frameworks such as Microsoft Cognitive Toolkit (CNTK), TensorFlow, Caffe2, MXNet, Keras, Theano, PyTorch and/or Chainer.

Please visit [here](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md) for detailed instruction.

Besides, we provide a one-click installer to setup all the frameworks automatically. Please follow below guidance if you want to use this installation tool.

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
