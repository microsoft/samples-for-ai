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


## How to run

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

### 3. Begin training!
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

### 5. Using main.py (optional)

We provide **main.py** for remote training for Visual Stdio or Visual Stdio Code users. So you can directely start the training process using command line or VS Debugger, which will be convenient in remote job submission.

```
$ python3 main.py 	--data_dir = <dataset h5 file directory> 	\
			--resume_dir = <weights snapshot directory> 	\
			--resume_kimg = <previous trained images in thousands>
```

## Contact us

Any bug report or advice, please contact us:

Kun Yan (naykun) : yankun1138283845@foxmail.com

Yihang Yin (Somedaywilldo) : somedaywilldo@foxmail.com

## License

Our code is under [MIT license](https://en.wikipedia.org/wiki/MIT_License). See [LICENSE](LICENSE)




