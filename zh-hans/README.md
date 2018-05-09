# 介绍

[English](/README.md)

通过Visual Studio解决方案的格式提供示例，让用户使用[Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai)入门深度学习。 每个解决方案有一个或多个示例项目。 解决方案是按照不同的深度学习框架组织的：

- CNTK (包含BrainScript和Python语言)
- Tensorflow
- Caffe2
- Keras
- MXNet
- Chainer
- Theano

# 贡献

本项目欢迎任何贡献和建议。 大多数贡献都需要你同意参与者许可协议（CLA），来声明你有权，并实际上授予我们有权使用你的贡献。 有关详细信息，请访问 https://cla.microsoft.com。

当你提交拉取请求时，CLA机器人会自动检查你是否需要提供CLA，并修饰这个拉取请求(例如，标签、注释)。 只需要按照机器人提供的说明进行操作即可。 CLA只需要通过一次，就能应用到所有的存储库上。

该项目采用了 [ Microsoft 开源行为准则 ](https://opensource.microsoft.com/codeofconduct/)。 有关详细信息,请参阅 [ 行为守则常见问题解答 ](https://opensource.microsoft.com/codeofconduct/faq/) 或联系<opencode@microsoft.com>咨询问题或评论。

# 准备工作

## 运行样例的先决条件

- 安装 [Microsoft Visual Studio](https://www.visualstudio.com/) 2017或2015。
- 安装[Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai)。
- 预下载数据 
    - 对于CNTK BrainScript MNIST项目，在"input"文件夹中，运行"python install_mnist.py"来下载数据。

## 准备开发环境

在本地或远程计算机上训练深度学习模型之前，确保安装了相应的深度学习软件。 如果有NVIDIA显卡，要安装最新的驱动和软件库。 还要确保安装了Python和必要的Python库，包括NumPy，SciPy，Python的Visual Studio支持，以及相应的深度学习框架，例如，微软认知工具包(CNTK)，TensorFlow, Caffe2, MXNet, Keras, Theano, PyTorch和Chainer。

请访问[这里](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md)获得更详细的说明。

## 一键式安装深度学习框架

当前，安装文件适用于Windows，macOS和Linux：

- 如果有NIVIDA显卡，需要安装最新的驱动，CUDA 9.0, 以及cuDNN 7.0。
- 安装最新的**Python 3.5或3.6**。 不支持其它Python版本。
- 在终端中运行以下命令：
    
    > [!注意]
    > 
    > - 如果Python安装在了系统目录中（如，与Visual Studio 2017一同发布的版本，或Linux的内置版本），则需要管理员权限（如，Linux下使用"sudo"）来运行安装脚本。
    > - 如果想安装到Python的用户目录中，可使用 "**--user**"参数。 通常在 ~/.local/，或windows下的%APPDATA%\Python目录。
    > - 安装程序会检测是否有可用的NVIDIA显卡，并默认安装CUDA 9.0软件。 可以使用"**--cuda80**"参数来强制安装CUDA 8.0 。
    
    ```bash
    git clone https://github.com/Microsoft/samples-for-ai.git
    cd samples-for-ai
    cd installer

    - Windows运行：
        python.exe install.py
    - 非Windows运行：
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

样例脚本来源于每个框架的官方github。 它们遵循不同的许可。

CNTK脚本遵循[MIT许可](https://en.wikipedia.org/wiki/MIT_License)。

Tensorflow脚本遵循[Apache 2.0许可](https://en.wikipedia.org/wiki/Apache_License#Version_2.0)。 在原始代码上没有改动。

对于Caffe2的脚本，不同的发布版本有不同的许可。 当前，主分支遵循Apache 2.0许可。 但版本0.7和0.8.1是在[BSD 2-Clause许可](https://github.com/caffe2/caffe2/tree/v0.8.1)下发布的。 我们的解决方案基于Caffe2的版本0.7和0.8.1的脚本源代码，遵循BSD 2-Clause许可。

Keras脚本遵循[MIT 许可](https://github.com/fchollet/keras/blob/master/LICENSE)。

Theano脚本遵循[BSD许可](https://en.wikipedia.org/wiki/BSD_licenses)。

MXNet脚本遵循[Apache 2.0许可](https://en.wikipedia.org/wiki/Apache_License#Version_2.0)。 在原始代码上没有改动。

Chainer脚本遵循[MIT许可](https://github.com/chainer/chainer/blob/master/LICENSE)。