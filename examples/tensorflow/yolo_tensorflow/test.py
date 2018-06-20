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

import numpy as np
import tensorflow as tf
import cv2

import config as cfg
from yolo_net.yolo_v1 import Yolo

def main():
    yolo = Yolo()

    ori_img = cv2.imread('test.jpg')
    height, width = ori_img.shape[:2]

    img = cv2.resize(ori_img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    img = np.expand_dims(img, axis=0)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint("checkpoint"))
    print(tf.train.latest_checkpoint("checkpoint"))
    preds = sess.run(yolo.net, feed_dict={yolo.images: img})[0] # [CELL_SIZE, CELL_SIZE, 5 * BOX_PER_CELL + CLASS_NUM]


    for i in range(cfg.CELL_SIZE):
        for j in range(cfg.CELL_SIZE):
            c = preds[i, j, 0:5*cfg.BOX_PER_CELL:5]
            c = np.expand_dims(c, axis=1)
            _class = preds[i, j, -cfg.CLASS_NUM:]
            _class = np.expand_dims(_class, axis=0)
            c_class = np.multiply(c, _class)

            max_c_class = np.max(c_class, axis=1)
            response = np.argmax(max_c_class)
            class_index = np.argmax(c_class[response])

            p = c_class[response][class_index]

            if p >= cfg.THRESHOLD:
                offset = 5 * response
                x = preds[i, j, 1+offset] * width
                y = preds[i, j, 2+offset] * height
                w = np.square(preds[i, j, 3+offset]) * width
                h = np.square(preds[i, j, 4+offset]) * height
                box = [
                    (int(x - w / 2), int(y - h / 2)),
                    (int(x + w / 2), int(y + h / 2))
                ]
                ori_img = cv2.rectangle(ori_img, box[0], box[1], (0, 255, 0), 2)

                class_name = cfg.CLASSES[class_index]
                cv2.putText(ori_img, class_name, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    cv2.imshow('image', ori_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()