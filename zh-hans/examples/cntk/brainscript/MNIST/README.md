# CNTK样例：MNIST

[English](/examples/cntk/brainscript/MNIST/README.md)

## 概述

| 数据： | MNIST手写体数据集(http://yann.lecun.com/exdb/mnist/)。          |
|:--- |:-------------------------------------------------------- |
| 目的  | 此样例演示了使用NDL (Network Description Language，网络描述语言) 来定义网络。 |
| 网络  | NDLNetworkBuilder，简单前馈，卷积网络，交叉熵与软最大值(SoftMax)。           |
| 训练  | 有动量(momentum) 和无动量的随机梯度下降。                               |
| 注释  | 有4个配置文件，详细信息如下。                                          |

## 运行样例

### 获取数据

MNIST数据集不包括在CNTK的分发包中，但能够很容易的被下载并通过运行'AdditionalFiles'目录中的命令来转换格式：

`python mnist_convert.py`

脚本会下载所有需要的文件，并转换为CNTK支持的格式。 结果文件 (Train-28x28_cntk_text.txt和Test-28x28_cntk_text.txt) 会存储在'Data'文件夹中。 如果还没有安装Python，则有两个选择：

1. 从 https://www.python.org/downloads/ 安装最新版的Python 2.7 然后通过以下指示来安装numpy包： http://www.scipy.org/install.html#individual-packages

2. 或者安装Python Anaconda分发包，它包含了numpy在内的大多数流行的Python包：http://continuum.io/downloads

### 设置

编译源文件来生成的CNTK可执行文件 (如果下载的是可执行文件，则请跳过这一步)。

**Windows:** 将CNTK可执行文件的文件夹加入到路径中 (例如 `set PATH=%PATH%;c:/src/cntk/x64/Debug/;`) 或者在调用CNTK可执行文件时，加上相应的路径。

**Linux:** 将CNTK可执行文件的文件夹加入到路径中 (例如 `export PATH=$PATH:$HOME/src/cntk/build/debug/bin/`) 或者在调用CNTK可执行文件时，加上相应的路径。

### 运行

用下面的命令来运行Image/MNIST/Data文件夹中的样例：

`cntk configFile=../Config/01_OneHidden_ndl_deprecated.cntk`

也可以从任意文件夹运行，并将`currentDirectory`指定到Data文件夹，例如从Image/MNIST目录运行时：

`cntk configFile=Config/01_OneHidden_ndl_deprecated.cntk currentDirectory=Data`

输出文件会创建在Image/MNIST/下。

## 详细说明

### 配置文件

'Config'文件夹中有4个配置文件和相应的网络描述文件：

1. 01_OneHidden.ndl 是简单的单层隐藏网络，它的错误率为2.3%。 要运行样例，打开Data文件夹，并运行此命令：  
    `cntk configFile=../Config/01_OneHidden_ndl_deprecated.cntk`

2. 02_Convolution.ndl 非常有意思。它有两个卷积层和两个最大池化层。 这个网络在GPU上训练大约两分钟后，错误率会降到0.87%。 要运行样例，打开Data文件夹，并运行此命令：  
    `cntk configFile=../Config/02_Convolution_ndl_deprecated.cntk`

3. 03_ConvBatchNorm.ndl与02_Convolution.ndl相比，除了它在卷积层和全连接层用了批量正则化的方法，几乎相同。 它在训练两个批次后（每次少于30秒），能达到大约0.8%的错误率。 要运行样例，打开Data文件夹并运行以下命令：  
    `cntk configFile=../Config/03_ConvBatchNorm_ndl_deprecated.cntk`

4. 04_DeConv.ndl展示了如何使用反卷积和反池化。 这个网络有一个卷积层，一个池化层，还有一个反池化层以及一个反卷积层。 实际上，它类似于一个自动编码网络。但自动编码网络的矫正线性单元（ReLU）或Sigmoid层被替换成了卷积ReLU层（用于编码）和反向卷积ReLU层（用于解码）。 这个网络的目标是用均方差（MSE）来重构原始信号，从而最小化重构的错误。 一般来说，这样的网络会用来做语义分割。  
    要运行样例，打开Data文件夹并运行以下命令：  
    `cntk configFile=../Config/04_DeConv_ndl_deprecated.cntk`

更多信息，参考.ndl文件和对应的.cntk文件。

### 其它文件

'AdditionalFiles'目录包含了下载和转换数据用的python脚本。