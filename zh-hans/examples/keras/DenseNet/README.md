# Keras中实现密集卷积网络(Dense Net)

DenseNet implementation of the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v3.pdf) in Keras

Now supports the more efficient DenseNet-BC (DenseNet-Bottleneck-Compressed) networks. Using the DenseNet-BC-190-40 model, it obtaines state of the art performance on CIFAR-10 and CIFAR-100

# 架构

DenseNet is an extention to Wide Residual Networks. According to the paper:   


    第l层有l个输入，与所有先前的卷积模块的特征映射一致。 
    而它自己的特征映射会被传递到所有L - l的后续层。 这样，L层上就会有 L * (L+1) /2 个连接，而不像传统的前馈网络一样，只有L个连接。 
    因为它的密集连接模式，我们将其称为密集卷积网络(DenseNet)。
    

It features several improvements such as :

1. 密集连接：每层都连接到其它所有层。
2. 增长率参数决定了，在网络深度增加时，特征数量的增长速度。
3. 连续函数：Wide ResNet 论文中的 BatchNorm - Relu - Conv，并根据ResNet论文进行了改进。

The Bottleneck - Compressed DenseNets offer further performance benefits, such as reduced number of parameters, with similar or better performance.

- DenseNet-100-12 模型有接近7百万个参数，而DenseNet-BC-100-1只有80万个参数。 原始模型的错误率为4.1%，而BC模型达到了4.51%的错误率。

- 原始的最好的模型 DenseNet-100-24 (2720万个参数) 达到了 3.74% 的错误率，而DenseNet-BC-190-40 (2560万个参数) 达到了3.46% 的错误率，这是CIFAR-10上的最好结果。

Dense Nets have an architecture which can be shown in the following image from the paper:   
<img src="https://github.com/titu1994/DenseNet/blob/master/images/dense_net.JPG?raw=true" />

# 性能

The accuracy of DenseNet has been provided in the paper, beating all previous benchmarks in CIFAR 10, CIFAR 100 and SVHN   
<img src="https://github.com/titu1994/DenseNet/blob/master/images/accuracy_densenet.JPG?raw=true" />

# 用法

Import the `densenet.py` script and use the `DenseNet(...)` method to create a custom DenseNet model with a variety of parameters.

Examples :

    import densenet
    
    # 'th' dim-ordering or 'tf' dim-ordering
    image_dim = (3, 32, 32) or image_dim = (32, 32, 3)
    
    model = densenet.DenseNet(classes=10, input_shape=image_dim, depth=40, growth_rate=12, 
                  bottleneck=True, reduction=0.5)
    

Or, Import a pre-built DenseNet model for ImageNet, with some of these models having pre-trained weights (121, 161 and 169).

Example :

    import densenet
    
    # 'th' dim-ordering or 'tf' dim-ordering
    image_dim = (3, 224, 224) or image_dim = (224, 224, 3)
    
    model = densenet.DenseNetImageNet121(input_shape=image_dim)
    

Weights for the DenseNetImageNet121, DenseNetImageNet161 and DenseNetImageNet169 models are provided ([in the release tab](https://github.com/titu1994/DenseNet/releases)) and will be automatically downloaded when first called. They have been trained on ImageNet. The weights were ported from the repository https://github.com/flyyufelix/DenseNet-Keras.

# 必需组件

- Keras
- Theano (权重未测试) / Tensorflow (已测试) / CNTK (权重未测试)
- h5Py