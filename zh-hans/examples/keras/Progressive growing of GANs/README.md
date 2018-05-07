# Keras-progressive_growing_of_gans

[简体中文](/zh-hans/examples/keras/Progressive%20growing%20of%20GANs/README.md)

## 介绍

Keras implementation of Progressive Growing of GANs for Improved Quality, Stability, and Variation.

Developed by BUAA Microsoft Student Club.

Leader Developers: Kun Yan, Yihang Yin, Xutong Li

Developers: Jiaqi Wang, Junjie Wu

## 必需组件

1. Python3 
2. keras 2.1.2 (TensorFlow后端)
3. [CelebA数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## 如何运行

### 1. 克隆代码库

### 2. 准备数据集

First download [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Run **h5tool.py** to create HDF5 format datatset. Default settings of **h5tool.py** will crop the picture to 128*128, and create a channel-last h5 file.

    $ python3 h5tool.py create_celeba_channel_last <h5 文件名> <CelebA 文件夹>
    

Modify **config.py** to your own settings.

    # In config.py:
    data_dir = 'datasets'
    result_dir = 'results'
    
    dataset = dict(h5_path=<h5 文件名>, resolution=128, max_labels=0, mirror_augment=True)
    # 注意: "data_dir" 应该被改为你的h5文件的文件夹
    

We only support CelebA dataset for now, you may need to modify the code in **dataset.py** and **h5tools.py** if you want to switch to another dataset.

### 3. 开始训练！

    $ python3 train.py
    

In **train.py**:

    # In train.py:
    speed_factor = 20
    # 如果不需要，可以设置为1。
    

"speed_factor" parameter will speed up the transition procedure of progressive growing of gans(switch resolution), at the price of reducing images' vividness, this parameter is aimed for speed up the validation progress in our development, however it is useful to see the progressive growing procedure more quickly, set it to "1" if you don't need it.

**So far, if your settings have no problem, you should see running information like our [running_log_example](running_log_example.txt)**

### 4. 保存并恢复训练权重

Parameters in **train.py** will determine the frequency of saving the training result snapshot. And if you want to resume a previous result, just modify **train.py**:

    # 在 train.py 中：
    image_grid_type         = 'default',
    # 将下面行
    # resume_network          = None,
    # 修改为：
    resume_network          = <权重快照的目录>,
    resume_kimg             = <以前训练的图片（千为单位）>,
    

### 5. 使用main.py (可选)

We provide **main.py** for remote training for Visual Studio or Visual Studio Code users. So you can directely start the training process using command line or VS Debugger, which will be convenient in remote job submission.

    $ python3 main.py   --data_dir = <数据集的h5文件目录>    \
                --resume_dir = <权重快照目录>     \
                --resume_kimg = <以前训练的图片（千为单位）>
    

## 结果

These two pictures are the training result we get so far, trained for 5 days on a NVIDIA GeForce 1080-ti GPU. You should be able to see the changes of resolution during the progressively growing procedure of our model. Pretrained weight will be available later.

![fakes003800](fakes003800.png)

![fakes008080](fakes008080.png)

## 联系我们

Any bug report or advice, please contact us:

Kun Yan (naykun) : yankun1138283845@foxmail.com

Yihang Yin (Somedaywilldo) : somedaywilldo@foxmail.com

## 参考

1. *Progressive Growing of GANs for Improved Quality, Stability, and Variation*, **Tero Karras** (NVIDIA), **Timo Aila** (NVIDIA), **Samuli Laine** (NVIDIA), **Jaakko Lehtinen** (NVIDIA and Aalto University) [Paper (NVIDIA research)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)

2. tkarras/progressive_growing_of_gans (https://github.com/tkarras/progressive_growing_of_gans

3. github-pengge/PyTorch-progressive_growing_of_gans (https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)

## 许可证

Our code is under [MIT license](https://en.wikipedia.org/wiki/MIT_License). See <LICENSE>

## 给中文用户的礼物

We have translated the original paper to Chinese briefly, a markdown version is available now, hope this will benefits chinese users.