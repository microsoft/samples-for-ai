import numpy as np
import sys
import os
import tensorflow as tf
from config import *
from classifier import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='0'

device = '/gpu:0'

def main(_):
    with tf.device(device):
      maybe_download_and_extract()
      # Initialize the Train object
      Cifar_Classifier = Classifier()
      # Start the training session
      Cifar_Classifier.train()
    exit(0)

if __name__ == "__main__":
    tf.app.run()