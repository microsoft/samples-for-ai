# How to use this YOLO model?

> This project is designed to reproduce YOLO model. 



## Require 

- Python 3.5 
- Tensorflow 1.5 or higher
- Opencv 3.0 or higher
- Numpy
- A pre-trained model

&emsp;&emsp; https://drive.google.com/file/d/0B-yiAeTLLamRekxqVE01Yi1RRlk/view?usp=sharing



## Test Demo

### Test by Command Line

```shell
cd yoloTF
python yoloTF.py
```

### Test using VS2017

You need to install VS2017, and configure it with tensorflow, python and opencv.

Open yoloTF project, find yoloTF.py file, and clikc "Run" button.

![](yoloTF/vs2017.png)

## Train 

##### Download pascal-Voc2007 data

1. Download the training, validation and test data

   ```
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar

   ```

3. It should have this basic structure

   ```
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. Create symlinks for the PASCAL VOC dataset

   ```
   cd $YOLO_ROOT/data
   ln -s $VOCdevkit VOCdevkit2007
   ```

   Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.

#### convert the Pascal-voc data to text_record file

```
python tools/preprocess_pascal_voc.py
```

#### train

```
python tools/train.py -c conf/train.cfg
```

#### Train your customer data

1. transform your training data to text_record file(the format reference to pascal_voc)
2. write your own train-configure file
3. train (python tools/train.py -c $your_configure_file)