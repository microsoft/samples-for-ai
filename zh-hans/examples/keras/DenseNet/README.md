# Keras中实现密集卷积网络(Dense Net)

[English](/examples/keras/DenseNet/README.md)

基于[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v3.pdf)的Keras中的DenseNet实现

现在已经支持了更高效的DenseNet-BC (DenseNet-Bottleneck-Compressed) 网络。 它使用了DenseNet-BC-190-40模型， 在CIFAR-10和CIFAR-100上获得了最高的性能。

# 架构

DenseNet是宽度残差网络（Wide Residual Networks）的扩展。 根据论文（下为译文，原文请参考英文版）：   


    第l层有l个输入，与所有先前的卷积模块的特征映射一致。 
    而它自己的特征映射会被传递到所有L - l的后续层。 这样，L层上就会有 L * (L+1) /2 个连接，而不像传统的前馈网络一样，只有L个连接。 
    因为它的密集连接模式，我们将其称为密集卷积网络(DenseNet)。
    

它的主要特点如下：

1. 密集连接：每层都连接到其它所有层。
2. 增长率参数决定了，在网络深度增加时，特征数量的增长速度。
3. 连续函数：Wide ResNet 论文中的 BatchNorm - Relu - Conv，并根据ResNet论文进行了改进。

Bottleneck - Compressed DenseNets 在相似甚至更好的结果下，进一步通过减少参数数量提高了性能。

- DenseNet-100-12 模型有接近7百万个参数，而DenseNet-BC-100-1只有80万个参数。 原始模型的错误率为4.1%，而BC模型达到了4.51%的错误率。

- 原始的最好的模型 DenseNet-100-24 (2720万个参数) 达到了 3.74% 的错误率，而DenseNet-BC-190-40 (2560万个参数) 达到了3.46% 的错误率，这是CIFAR-10上的最好结果。

该论文中的下图描述了密集网络的架构：   
<img src="https://github.com/titu1994/DenseNet/blob/master/images/dense_net.JPG?raw=true" />

# 性能

论文中提供了DenseNet的精度，并超过了所有在CIFAR 10, CIFAR 100和SVHN上的基准算法。   
<img src="https://github.com/titu1994/DenseNet/blob/master/images/accuracy_densenet.JPG?raw=true" />

# 用法

导入`densenet.py`脚本，并使用`DenseNet(...)`方法来通过参数组合来创建一个自定义的DenseNet模型。

样例：

    import densenet
    
    # 'th' dim-ordering or 'tf' dim-ordering
    image_dim = (3, 32, 32) or image_dim = (32, 32, 3)
    
    model = densenet.DenseNet(classes=10, input_shape=image_dim, depth=40, growth_rate=12, 
                  bottleneck=True, reduction=0.5)
    

或者，从ImageNet导入已有的DenseNet模型，一些模型已经有了预训练的权重(121, 161和169)。

样例：

    import densenet
    
    # 'th' dim-ordering or 'tf' dim-ordering
    image_dim = (3, 224, 224) or image_dim = (224, 224, 3)
    
    model = densenet.DenseNetImageNet121(input_shape=image_dim)
    

DenseNetImageNet121, DenseNetImageNet161和DenseNetImageNet169模型的权重已提供在 ([发布标签页](https://github.com/titu1994/DenseNet/releases))，会在第一次调用时下载。 它们是在ImageNet数据集上训练的。 这些权重也移到了代码库 https://github.com/flyyufelix/DenseNet-Keras 。

# 必需组件

- Keras
- Theano (权重未测试) / Tensorflow (已测试) / CNTK (权重未测试)
- h5Py