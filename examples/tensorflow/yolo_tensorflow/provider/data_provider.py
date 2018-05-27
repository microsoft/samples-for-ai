from os import listdir
from os.path import isfile, join

import cv2
import xml.etree.cElementTree as ET
import numpy as np

import config as cfg

import os


class DataProvider(object):

    def __init__(self):
        self.jpeg_dir = cfg.DATA_PATH + 'JPEGImages/'
        self.label_dir = cfg.DATA_PATH + 'Annotations/'

        self.cursor = 0
        self.epoch = 1
        self.gl_labels = []

        self._init_data()

    def _init_data(self):

        for file_name in listdir(self.jpeg_dir):
            root = ET.ElementTree(file=self.label_dir + file_name.replace("jpg", "xml")).getroot()
            size_ele = root.find('size')
            width = float(size_ele.find('width').text)
            height = float(size_ele.find('height').text)
            w_scale = cfg.IMAGE_SIZE / width
            h_scale = cfg.IMAGE_SIZE / height

            data = np.zeros([cfg.CELL_SIZE, cfg.CELL_SIZE, 5 + cfg.CLASS_NUM], np.float32)

            for obj_ele in root.findall('object'):
                class_name = obj_ele.find('name').text
                class_index = cfg.CLASSES.index(class_name)
                bndbox = obj_ele.find('bndbox')
                xmin = (float(bndbox.find('xmin').text) - 1) * w_scale
                ymin = (float(bndbox.find('ymin').text) - 1) * h_scale
                xmax = (float(bndbox.find('xmax').text) - 1) * w_scale
                ymax = (float(bndbox.find('ymax').text) - 1) * h_scale

                boxes = [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]
                x_index = int(boxes[0] * cfg.CELL_SIZE / cfg.IMAGE_SIZE)
                y_index = int(boxes[1] * cfg.CELL_SIZE / cfg.IMAGE_SIZE)

                boxes = [value / cfg.IMAGE_SIZE for value in boxes]

                if data[y_index, x_index, 0] == 1:
                    continue

                data[y_index, x_index, 0] = 1
                data[y_index, x_index, 1:5] = boxes
                data[y_index, x_index, 5 + class_index] = 1

            label = {'imname': file_name, 'data': data}
            self.gl_labels.append(label)

        self.train_size = len(self.gl_labels)

    def get_data(self):
        """Get train data

        Returns:
            images: [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]
            labels: [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 5 + CLASS_NUM]
        """

        images = np.zeros((cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), np.float32)
        labels = np.zeros((cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 5 + cfg.CLASS_NUM), np.float32)

        for i in range(cfg.BATCH_SIZE):
            label = self.gl_labels[self.cursor]
            self.cursor += 1

            if self.cursor == self.train_size:
                self.cursor = 0
                self.epoch += 1

            impath = self.jpeg_dir + label['imname']
            image = cv2.imread(impath)
            image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

            images[i] = image
            labels[i] = label['data']

        return images, labels
