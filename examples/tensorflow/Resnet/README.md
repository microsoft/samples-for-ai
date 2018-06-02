# ResNet

This implementation of [Resnet:Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) is designed to be straightforward and friendly to Resnet freshman. 

The official implementation is avaliable at [tensorflow/model](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py).The official implementation of object detection is also released at [tensorflow/model/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection).

# Visual Studio Tools for AI 
This implementation is built on Visual Studio Tools for AI, which is a free Visual Studio extension to build, test, and deploy deep learning / AI solutions.More information in [VS Tools for AI](https://github.com/Microsoft/vs-tools-for-ai).

Requirements: python3 and tensorflow(1.7.0). Tested on Windows 7 with GTX 750. 


## Usage

### Train on CIFAR-10

1. Prepare data. You can modify ```cifar10_input.py``` according to your environment, then use it to download and convert CIFAR-10.
2. There are screen outputs, tensorboard statistics and tensorboard graph visualization to help you monitor the training process and visualize the model.

### Evaluate on CIFAR-10
1. Modify the configure variables, like ```test_ckpt_path``` in ```config.py``` according to your environment.

## Accuracy on CIFAR-10 Set

| Model | dataset | Accuracy-Top1 |
|--------|:--------|:---------:|
| ResNet_110 |train | 95.32% |
| ResNet_110 |validate | 86.40% |

## Pretrained weight 
A Pretrained weight see ```./pretrained```.

**train_top1_error**
<div align="center">
<img src=https://github.com/SugarMasuo/4-seu-AIGO/blob/master/ResNet-on-Tensorflow/result/train_top1_error.png"><br><br>
</div>

**validate_top1_error**
<div align="center">
<img src="https://github.com/SugarMasuo/4-seu-AIGO/blob/master/ResNet-on-Tensorflow/result/validate_top1_error.png"><br><br>
</div>



## what's next
- [ ] Call for more powerful GPU ot train on KITTI, Imagenet
- [ ] Try more optimizers like YellowFin
- [ ] Object detection task
- [ ] multi-gpu performance problems



## Reference
[Resnet](http://arxiv.org/abs/1512.03385)

[Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai)

[models-Tensorflow](https://github.com/tensorflow/models/tree/master/research)
