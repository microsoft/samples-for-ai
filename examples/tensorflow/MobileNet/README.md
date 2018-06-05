# MobileNet

This implementation of [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) is designed to be straightforward and friendly to MobileNet freshman. 

1. MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. 
2. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used.
3. The official implementation is avaliable at [tensorflow/model](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).The official implementation of object detection is also released at [tensorflow/model/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection).

# Visual Studio Tools for AI 
This implementation is built on Visual Studio Tools for AI, which is a free Visual Studio extension to build, test, and deploy deep learning / AI solutions.More information in [VS Tools for AI](https://github.com/Microsoft/vs-tools-for-ai).

Requirements: python3 and tensorflow(1.7.0). Tested on Windows 7 with GTX 750. 


## Usage

### Train on CIFAR-10

1. Prepare data. You can modify ```data_dir``` in ```download_and_convert_cifar10.py``` according to your environment, then use it to download CIFAR-10 and convert to TFRecord.
2. Modify ```dataset_dir``` in ```train_image_classifier``` according to your environment.
3. There are screen outputs, tensorboard statistics and tensorboard graph visualization to help you monitor the training process and visualize the model.

### Evaluate on CIFAR-10
1. Modify ```dataset_dir``` in ```train_image_classifier``` according to your environment.

## Accuracy on CIFAR-10 Set

| Model | dataset | Width Multiplier |Preprocessing  | Accuracy-Top1 |
|--------|:--------|:---------:|:------:|:------:|
| MobileNet |train|1.0| Same as Inception | 89.71% |
| MobileNet |test |1.0| Same as Inception | 85.83% |

## Pretrained weight 
A Pretrained weight was trained on GTX 750.
See ```./pretrained```.

**Loss**
<div align="center">
<img src="https://github.com/SugarMasuo/4-seu-AIGO/blob/master/MobileNet-on-tensorflow/result/loss.png"><br><br>
</div>



## what's next
- [ ] Call for more powerful GPU ot train on KITTI, Imagenet
- [ ] Try more optimizers like YellowFin
- [ ] Object detection task
- [ ] multi-gpu performance problems



## Reference
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai)

[models-Tensorflow](https://github.com/tensorflow/models/tree/master/research)
