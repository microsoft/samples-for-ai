# GeneGAN: 从未配对的数据中学习对象变形和属性子空间

[简体中文](/zh-hans/examples/tensorflow/GeneGAN/README.md)

By Shuchang Zhou, Taihong Xiao, Yi Yang, Dieqiao Feng, Qinyao He, Weiran He

If you use this code for your research, please cite our paper:

    @inproceedings{DBLP:conf/bmvc/ZhouXYFHH17,
      author    = {Shuchang Zhou and
                   Taihong Xiao and
                   Yi Yang and
                   Dieqiao Feng and
                   Qinyao He and
                   Weiran He},
      title     = {GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data},
      booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
      year      = {2017},
      url       = {http://arxiv.org/abs/1705.04932},
      timestamp = {http://dblp.uni-trier.de/rec/bib/journals/corr/ZhouXYFHH17},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
    

### 介绍

This is the official source code for the paper [GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data](https://arxiv.org/abs/1705.04932v1). All the experiments are initially done in our proprietary deep learning framework. For convenience, we reproduce the results using TensorFlow.

<div align="center">
<img align="center" src="images/cross.jpg" width="450" alt="交叉">
</div>

  


GeneGAN is a deterministic conditional generative model that can learn to disentangle the object features from other factors in feature space from weak supervised 0/1 labeling of training data. It allows fine-grained control of generated images on a certain attribute in a continous way.

### 必需组件

- Python 3.5
- TensorFlow 1.0
- Opencv 3.2

### 在celebA数据集上训练GeneGAN

1. 下载[celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)数据集，并解压缩到 `datasets`文件夹中。 CelebA数据集提供了多种源格式。 为了确保下载的图片尺寸是正确的，运行 `identify datasets/celebA/data/000001.jpg`。 正确的数据集大小应该是409*687。 此外，确保目录结构和下面的一致。

    ├── datasets
    │   └── celebA
    │       ├── data
    │       ├── list_attr_celeba.txt
    │       └── list_landmarks_celeba.txt
    

1. 运行`python preprocess.py`。 它会花几分钟来预处理所有人脸图片。 然后会创建新文件夹`datasets/celebA/align_5p`。

2. 运行`python train.py -a Bangs -g 0`在属性`Bangs`上训练GeneGAN。 也可以在其它属性上训练GeneGAN。 所有可用的属性都列在了`list_attr_celeba.txt`文件中。

3. 运行`tensorboard --logdir='./' --port 6006`来查看训练进度。

### 测试

We provide three kinds of mode for test. Run `python test.py -h` for detailed help. The following example is running on our GeneGAN model trained on the attribute `Bangs`. Have fun!

#### 1. 交换属性

You can easily add the bangs of one person to another person without bangs by running

    python test.py -m swap -i datasets/celebA/align_5p/182929.jpg -t datasets/celebA/align_5p/022344.jpg
    

<div align="center">
  <img align="center" src="images/182929_resize.jpg" alt="输入" /> <img align="center" src="images/022344_resize.jpg" alt="目标" /> <img align="center" src="images/swap_out1.jpg" alt="输出1" /> <img align="center" src="images/swap_out2.jpg" alt="输出2" />
</div>

<div align="center">
交换属性
</div>

  


#### 2. 图像属性的线性插值

Besides, we can control to which extent the bangs style is added to your input image through linear interpolation of image attribute. Run the following code.

    python test.py -m interpolation -i datasets/celebA/align_5p/182929.jpg -t datasets/celebA/align_5p/035460.jpg -n 5
    

<div align="center">
  <img align="center" src="images/interpolation.jpg" alt="插值" /> <img align="center" src="images/035460_resize.jpg" alt="目标" />
</div>

<div align="center">
线性插值
</div>

  


#### 3. 属性子空间中的矩阵插值

We can do something cooler. Given four images with bangs attributes at hand, we can observe the gradual change process of our input images with a mixing of difference bangs style.

    python test.py -m matrix -i datasets/celebA/align_5p/182929.jpg --targets datasets/celebA/align_5p/035460.jpg datasets/celebA/align_5p/035451.jpg datasets/celebA/align_5p/035463.jpg datasets/celebA/align_5p/035474.jpg -s 5 5
    

<div align="center">
<img align="center" src="images/four_matrix.jpg" alt="矩阵">
</div>

<div align="center">
矩阵插值
</div>

