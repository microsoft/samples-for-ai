# CNTK样例：MNIST

[简体中文](/zh-hans/examples/cntk/brainscript/MNIST/README.md)

## 概述

| 数据： | MNIST手写体数据集(http://yann.lecun.com/exdb/mnist/)。          |
|:--- |:-------------------------------------------------------- |
| 目的  | 此样例演示了使用NDL (Network Description Language，网络描述语言) 来定义网络。 |
| 网络  | NDLNetworkBuilder，简单前馈，卷积网络，交叉熵与软最大值(SoftMax)。           |
| 训练  | 有动量(momentum) 和无动量的随机梯度下降。                               |
| 注释  | 有4个配置文件，详细信息如下。                                          |

## 运行样例

### 获取数据

The MNIST dataset is not included in the CNTK distribution but can be easily downloaded and converted by running the following command from the 'AdditionalFiles' folder:

`python mnist_convert.py`

The script will download all required files and convert them to CNTK-supported format. The resulting files (Train-28x28_cntk_text.txt and Test-28x28_cntk_text.txt) will be stored in the 'Data' folder. In case you don't have Python installed, there are 2 options:

1. 从 https://www.python.org/downloads/ 安装最新版的Python 2.7 然后通过以下指示来安装numpy包： http://www.scipy.org/install.html#individual-packages

2. 或者安装Python Anaconda分发包，它包含了numpy在内的大多数流行的Python包：http://continuum.io/downloads

### 设置

Compile the sources to generate the cntk executable (not required if you downloaded the binaries).

**Windows:** Add the folder of the cntk executable to your path (e.g. `set PATH=%PATH%;c:/src/cntk/x64/Debug/;`) or prefix the call to the cntk executable with the corresponding folder.

**Linux:** Add the folder of the cntk executable to your path (e.g. `export PATH=$PATH:$HOME/src/cntk/build/debug/bin/`) or prefix the call to the cntk executable with the corresponding folder.

### 运行

Run the example from the Image/MNIST/Data folder using:

`cntk configFile=../Config/01_OneHidden_ndl_deprecated.cntk`

or run from any folder and specify the Data folder as the `currentDirectory`, e.g. running from the Image/MNIST folder using:

`cntk configFile=Config/01_OneHidden_ndl_deprecated.cntk currentDirectory=Data`

The output folder will be created inside Image/MNIST/.

## 详细说明

### 配置文件

There are four config files and the corresponding network description files in the 'Config' folder:

1. 01_OneHidden.ndl 是简单的单层隐藏网络，它的错误率为2.3%。 要运行样例，打开Data文件夹，并运行此命令：  
    `cntk configFile=../Config/01_OneHidden_ndl_deprecated.cntk`

2. 02_Convolution.ndl 非常有意思。它有两个卷积层和两个最大池化层。 这个网络在GPU上训练大约两分钟后，错误率会降到0.87%。 要运行样例，打开Data文件夹，并运行此命令：  
    `cntk configFile=../Config/02_Convolution_ndl_deprecated.cntk`

3. 03_ConvBatchNorm.ndl与02_Convolution.ndl相比，除了它在卷积层和全连接层用了批量正则化的方法，几乎相同。 它在训练两个批次后（每次少于30秒），能达到大约0.8%的错误率。 要运行样例，打开Data文件夹并运行以下命令：  
    `cntk configFile=../Config/03_ConvBatchNorm_ndl_deprecated.cntk`

4. 04_DeConv.ndl展示了如何使用反卷积和反池化。 这个网络有一个卷积层，一个池化层，还有一个反池化层以及一个反卷积层。 实际上，它类似于一个自动编码网络。但自动编码网络的矫正线性单元（ReLU）或Sigmoid层被替换成了卷积ReLU层（用于编码）和反向卷积ReLU层（用于解码）。 这个网络的目标是用均方差（MSE）来重构原始信号，从而最小化重构的错误。 一般来说，这样的网络会用来做语义分割。  
    要运行样例，打开Data文件夹并运行以下命令：  
    `cntk configFile=../Config/04_DeConv_ndl_deprecated.cntk`

For more details, refer to .ndl and the corresponding .cntk files.

### 其它文件

The 'AdditionalFiles' folder contains the python script to download and convert the data.