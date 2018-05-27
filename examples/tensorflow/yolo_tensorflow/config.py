
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
