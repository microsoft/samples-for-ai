# Keras-progressive_growing_of_gans

## Introduction

Keras implementation of Progressive Growing of GANs for Improved Quality, Stability, and Variation. 

Developed by BUAA Microsoft Student Club.

Leader Developers: Kun Yan, Yihang Yin, Xutong Li

Developers: Jiaqi Wang, Junjie Wu


## Requirements

1. Python3 
2. keras 2.1.2 (TensorFlow backend)
3. [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


## How to run (For Visual Studio/ VSCode Users)

### 1. Clone the repository

### 2. Open it as Visual Studio solution

#### To train

1. Set **main.py** as the Startup File. 
2. Right click **main.py**, click "Start without Debugging" or "Start with Debugging" context menus.
3. Please read the "Prepare the dataset" carefully if you had dataset issues.

### 3. Prepare the dataset 

1. If you're able to access Google Drive, when you run **main.py**, it will automatically download the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (1.5GB) to **/datasets** directory and unzip it to **/datasets/CelebA** directory.
2. If you can not access Google Drive, please download the  [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (It has Baidu Cloud Disk link), notice that we only need the **"img_align_celeba.zip"** file.  Unzip it to **/datasets/CelebA** directory. Then when you run **main.py** it will skip the download process and begin training.

**Notice: /datasets/CelebA** should just contains 202599 pictures.

### 4. Save and resume training weights

Parameters in **train.py** will determine the frequency of saving the training result snapshot. And if you want to resume a previous result, just modify **train.py**:

```
# In train.py:
image_grid_type         = 'default',
# modify this line bellow
# resume_network          = None,
# to:
resume_network          = <weights snapshot directory>,
resume_kimg             = <previous trained images in thousands>,
```



## How to run (For other Users)

### 1. Clone the repository

### 2. Prepare the dataset

First download [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Run **h5tool.py** to create HDF5 format datatset. Default settings of **h5tool.py** will crop the picture to 128*128, and create a channel-last h5 file.

```
$ python3 h5tool.py create_celeba_channel_last <h5 file name> <CelebA directory>
```

Modify **config.py** to your own settings.

```
# In config.py:
data_dir = 'datasets'
result_dir = 'results'

dataset = dict(h5_path=<h5 file name>, resolution=128, max_labels=0, mirror_augment=True)
# Note: "data_dir" should be set to the direcory of your h5 file.
```

We only support CelebA dataset for now, you may need to modify the code in **dataset.py** and **h5tools.py** if you want to switch to another dataset.

### 3. Begin training
```
$ python3 train.py
```

In **train.py**:

```
# In train.py:
speed_factor = 20
# set it to 1 if you don't need it.
```

"speed_factor" parameter will speed up the transition procedure of progressive growing of gans(switch resolution), at the price of reducing images' vividness, this parameter is aimed for speed up the validation progress in our development, however it is useful to see the progressive growing procedure more quickly, set it to "1" if you don't need it.

### 4.Save and resume training weights

The operations are the same as VStdio/VSCode users.



**So far, if your settings have no problem, you should see running information like our [running_log_example](Example/running_log_example.txt)**



## Results

These two pictures are the training result we get so far, trained for 5 days on a NVIDIA GeForce 1080-ti GPU. You should be able to see the changes of resolution during the  progressively growing procedure of our model. 

![fakes003800](Example/fakes003800.png)

![fakes008080](Example/fakes008080.png)

## Pre-trained weight

We provide a set of pre-trained weights. Available at [Google Drive](https://drive.google.com/open?id=1c1YrBmwJ6b_ovoLY2E81wYlTRbyM7BHd) and [Baidu Cloud Disk](https://pan.baidu.com/s/1NbclLikgQOKpTXq7j8JI4w#list/path=%2F).

There are two ways to use the pretrained weight:

1. Use it as trained network to resume training.
2. Use it to generate fake CelebA faces directly.

For the second purpose, we provide **predict.py**, download the weights and put it **/pre-trained_weight** directory (Under the same directory as **/datasets** )

## Contact us

Any bug report or advice, please contact us:

Kun Yan (naykun) : yankun1138283845@foxmail.com

Yihang Yin (Somedaywilldo) : somedaywilldo@foxmail.com

## Reference

1. *Progressive Growing of GANs for Improved Quality, Stability, and Variation*, **Tero Karras** (NVIDIA), **Timo Aila** (NVIDIA), **Samuli Laine** (NVIDIA), **Jaakko Lehtinen** (NVIDIA and Aalto University) [Paper (NVIDIA research)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)

2. tkarras/progressive_growing_of_gans (https://github.com/tkarras/progressive_growing_of_gans

2. github-pengge/PyTorch-progressive_growing_of_gans (https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)

## License

Our code is under [MIT license](https://en.wikipedia.org/wiki/MIT_License). See [LICENSE](LICENSE)

## A Gift for Chinese Users

We have translated the original paper to Chinese briefly, a markdown version is available now, hope this will benefits chinese users.