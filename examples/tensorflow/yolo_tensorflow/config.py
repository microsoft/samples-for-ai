# MIT License
#
# Copyright (c) 2018 luoyi,kanxuan,dingyusheng,cuihejie,liyuan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

DATA_PATH = 'VOCdevkit/VOC2007/'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

CLASS_NUM = len(CLASSES)

#
# solver parameter
#

MAX_ITER = 50000

SUMMARY_ITER = 1

SAVE_ITER = 1000   # 1000

LEARNING_RATE = 0.0001  # 0.0001

DECAY_STEPS = 10000

DECAY_RATE = 0.1

BATCH_SIZE = 16

STAIRCASE = True

#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOX_PER_CELL = 2

COORD_SCALE = 5.0

NOOBJ_SCALE = 0.5

#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
