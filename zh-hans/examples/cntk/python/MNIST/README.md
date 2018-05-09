# MNIST 数据集

[English](/examples/cntk/python/MNIST/README.md)

MINST数字手写体 (http://yann.lecun.com/exdb/mnist/) 是试验各种分类算法最常用的数据集之一。 MNIST有6万个训练样本和1万个测试样本。 每个样例里只有一个数字，大小基本一致并在中间，并进行了灰度化，分辨率为28*28。

MNIST数据集没有包含在CNTK发布包里，但它能很容易的下载，转换为CNTK支持的格式。进入目录 Examples/Image/DataSets/MNIST，并运行以下Python命令：

`python install_mnist.py`

运行完命令后，会在当前目录看到两个输出文件：`Train-28x28_cntk_text.txt`和`Test-28x28_cntk_text.txt`。 需要大约`124`MB的磁盘空间。 现在可以进入[`GettingStarted`](../../GettingStarted)文件夹来使用这个数据集。

此外，我们还提供了两个MNIST的高级样例。 第一个是[`多层感知网络 (MLP)`](../../Classification/MLP)，能达到大约1.5%的错误率。 第二个是[`卷积神经网络 (ConvNet)`](../../Classification/ConvNet)，能达到大约0.5%的错误率。 这些结果的质量能与公开的用这些网络的最好结果相媲美。

如果想知道现在计算机在MNIST上的表现如何，Rodrigo Benenson的[博客](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)上有各种算法最新的性能。