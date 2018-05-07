# 介绍

[简体中文](zh-hans/README.md)

Samples in Visual Studio solution format are provided for users to get started with deep learning using [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai). Each solution has one or more sample projects. Solutions are separated by different deep learning frameworks they use:

- CNTK (包含BrainScript和Python语言)
- Tensorflow
- Caffe2
- Keras
- MXNet
- Chainer
- Theano

# 贡献

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact <opencode@microsoft.com> with any additional questions or comments.

# 准备工作

## 运行样例的先决条件

- 安装 [Microsoft Visual Studio](https://www.visualstudio.com/) 2017或2015。
- 安装[Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai)。
- 预下载数据 
    - 对于CNTK BrainScript MNIST项目，在"input"文件夹中，运行"python install_mnist.py"来下载数据。

## 准备开发环境

Before training deep learning models on your local or remote computer, please make sure you have the deep learning software installed. This includes the latest drivers and libraries for your NVIDIA GPU (if you have one). You also need to install Python and libraries such as NumPy, SciPy, Python support for Visual Studio, and frameworks such as Microsoft Cognitive Toolkit (CNTK), TensorFlow, Caffe2, MXNet, Keras, Theano, PyTorch and/or Chainer.

Please visit [here](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md) for detailed instruction.

## 一键式安装深度学习框架

Currently, this installer works on Windows, macOS and Linux:

- Install latest NVIDIA GPU driver, CUDA 9.0, and cuDNN 7.0 if applicable.
- 安装最新的**Python 3.5或3.6**。 不支持其它Python版本。
- Run the following commands in a terminal:
    
    > [!NOTE]
    > 
    > - If your Python distribution is installed in the system directory (e.g. the one shipped with Visual Studio 2017, or the built-in one on Linux), administrative permission (e.g. "sudo" on Linux) is required to launch the installer.
    > - Pass "**--user**" argument, if you want to install to the Python user install directory for your platform. Typically ~/.local/, or %APPDATA%\Python on Windows.
    > - The installer will detect whether NVIDIA GPU cards are available and set up software for CUDA 9.0 by default. You can pass "**--cuda80**" argument to force installing software for CUDA 8.0 .
    
    ```bash
git clone https://github.com/Microsoft/samples-for-ai.git
cd samples-for-ai
cd installer

- Windows:
    python.exe install.py
- Non-Windows:
    python3 install.py
```

## 在本地运行样例

- CNTK BrainScript 项目
    
    - 将想要运行的项目设置为"启动项目"。
    - 将想要运行的脚本设置为"Startup File"。
    - 点击"Run CNTK Brain Script"。

- Python 项目
    
    - 设置"启动文件"。
    - 右击Python启动脚本，并点击上下文菜单中的"在不调试的情况下启动"或"开始执行(调试)"。

# 许可证

The samples scripts are from official github of each framework. They are under different licenses.

The scripts of CNTK are under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

The scripts of Tensorflow samples are under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0). There are no changes on the original code.

For the scripts of Caffe2, different versions released with different licenses. Currently, the master branch is under Apache 2.0 license. But the version 0.7 and 0.8.1 were released with [BSD 2-Clause license](https://github.com/caffe2/caffe2/tree/v0.8.1). The scripts in our solution are based on caffe2 github source tree version 0.7 and 0.8.1, with BSD 2-Clause license.

The scripts of Keras are under [MIT license](https://github.com/fchollet/keras/blob/master/LICENSE).

The scripts of Theano are under [BSD license](https://en.wikipedia.org/wiki/BSD_licenses).

The scripts of MXNet are under [Apache 2.0 license](https://en.wikipedia.org/wiki/Apache_License#Version_2.0). There are no changes on the original code.

The scripts of Chainer are under [MIT license](https://github.com/chainer/chainer/blob/master/LICENSE).