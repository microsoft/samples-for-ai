# GeneGAN: 从未配对的数据中学习对象变形和属性子空间

[English](/examples/tensorflow/GeneGAN/README.md)

作者：Shuchang Zhou, Taihong Xiao, Yi Yang, Dieqiao Feng, Qinyao He, Weiran He

如果在研究里使用了这些代码，请引用此论文：

```
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
```

### 介绍

官方的源代码来源于论文：[GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data](https://arxiv.org/abs/1705.04932v1)。 所有试验最初都在我们专门的深度学习框架中完成。 为了方便使用，我们用Tensorflow重现了结果。

<div align="center">
<img align="center" src="images/cross.jpg" width="450" alt="交叉">
</div>

  


GeneGAN是一种确定性条件生成模型，它从训练数据的弱监督分类的0/1标记中学习，将目标特征从特征空间的其它因素里分离出来。 它允许在某个属性上连续的控制生成图像的细节颗粒度。

### 必需组件

- Python 3.5
- TensorFlow 1.0
- Opencv 3.2

### 在celebA数据集上训练GeneGAN

1. 下载[celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)数据集，并解压缩到 `datasets`文件夹中。 CelebA数据集提供了多种源格式。 为了确保下载的图片尺寸是正确的，运行 `identify datasets/celebA/data/000001.jpg`。 正确的数据集大小应该是409*687。 此外，确保目录结构和下面的一致。

```
├── datasets
│   └── celebA
│       ├── data
│       ├── list_attr_celeba.txt
│       └── list_landmarks_celeba.txt
```

1. 运行`python preprocess.py`。 它会花几分钟来预处理所有人脸图片。 然后会创建新文件夹`datasets/celebA/align_5p`。

2. 运行`python train.py -a Bangs -g 0`在属性`Bangs`上训练GeneGAN。 也可以在其它属性上训练GeneGAN。 所有可用的属性都列在了`list_attr_celeba.txt`文件中。

3. 运行`tensorboard --logdir='./' --port 6006`来查看训练进度。

### 测试

有三种测试模式。 运行`python test.py -h`查看详细帮助。 下面的样例是在`Bangs`属性上训练的GeneGAN模型。 玩得高兴！

#### 1. 交换属性

运行以下命令来将一个人的刘海添加到另一个人上

    python test.py -m swap -i datasets/celebA/align_5p/182929.jpg -t datasets/celebA/align_5p/022344.jpg
    

<div align="center">
  <img align="center" src="images/182929_resize.jpg" alt="输入" /> <img align="center" src="images/022344_resize.jpg" alt="目标" /> <img align="center" src="images/swap_out1.jpg" alt="输出1" /> <img align="center" src="images/swap_out2.jpg" alt="输出2" />
</div>

<div align="center">
交换属性
</div>

  


#### 2. 图像属性的线性插值

此外，通过图像属性的线性插值，还能控制刘海样式的程度。 运行下列代码。

    python test.py -m interpolation -i datasets/celebA/align_5p/182929.jpg -t datasets/celebA/align_5p/035460.jpg -n 5
    

<div align="center">
  <img align="center" src="images/interpolation.jpg" alt="插值" /> <img align="center" src="images/035460_resize.jpg" alt="目标" />
</div>

<div align="center">
线性插值
</div>

  


#### 3. 属性子空间中的矩阵插值

还有更酷的操作。 同时给出4个有刘海属性的图像，能观察到混合了不同刘海样式的渐近变化过程。

    python test.py -m matrix -i datasets/celebA/align_5p/182929.jpg --targets datasets/celebA/align_5p/035460.jpg datasets/celebA/align_5p/035451.jpg datasets/celebA/align_5p/035463.jpg datasets/celebA/align_5p/035474.jpg -s 5 5
    

<div align="center">
<img align="center" src="images/four_matrix.jpg" alt="矩阵">
</div>

<div align="center">
矩阵插值
</div>

