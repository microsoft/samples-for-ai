import numpy as np
import re
import itertools
from collections import Counter
import xml.dom.minidom
import jieba

def clean_str(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_data(path):
    dom = xml.dom.minidom.parse(path)
    TestingData = dom.documentElement
    weibos = TestingData.getElementsByTagName("weibo")
    x_data, y_data = [], []
    for item in weibos:
        emotion = item.getAttribute("emotion-type1")
        #print(emotion)
        tempSent = ''
        if len(emotion) > 0 and emotion!="无":
            sentences = item.getElementsByTagName('sentence')
            for sentence in sentences:
                tempSent += sentence.childNodes[0].data
            tempSent = clean_str(tempSent)
            tokens = list(jieba.cut(tempSent))
            x = ' '
            temp = x.join(tokens)
            x_data.append(temp)
            if emotion == '恐惧':
                y_data.append([1, 0, 0, 0, 0, 0, 0])
            elif emotion == '愤怒':
                y_data.append([0, 1, 0, 0, 0, 0, 0])
            elif emotion == '厌恶':
                y_data.append([0, 0, 1, 0, 0, 0, 0])
            elif emotion == '悲伤':
                y_data.append([0, 0, 0, 1, 0, 0, 0])
            elif emotion == '高兴':
                y_data.append([0, 0, 0, 0, 1, 0, 0])
            elif emotion == '喜好':
                y_data.append([0, 0, 0, 0, 0, 1, 0])
            elif emotion == '惊讶':
                y_data.append([0, 0, 0, 0, 0, 0, 1])
            else:
                print('error')
    y_data = np.array(y_data)
    print(y_data.shape)
    return x_data, y_data

def load_data_(path):
    dom = xml.dom.minidom.parse(path)
    TestingData = dom.documentElement
    weibos = TestingData.getElementsByTagName("weibo")
    x_data, y_data = [], []
    for item in weibos:
        emotion = item.getAttribute("emotion-type")
        tempSent = ''
        if len(emotion) > 0 and emotion != "无" and emotion != 'none':
            sentences = item.getElementsByTagName('sentence')
            for sentence in sentences:
                tempSent += sentence.childNodes[0].data
            tempSent = clean_str(tempSent)
            tokens = list(jieba.cut(tempSent))
            x = ' '
            temp = x.join(tokens)
            x_data.append(temp)
            if emotion == 'fear':
                y_data.append([1, 0, 0, 0, 0, 0, 0])
            elif emotion == 'anger':
                y_data.append([0, 1, 0, 0, 0, 0, 0])
            elif emotion == 'disgust':
                y_data.append([0, 0, 1, 0, 0, 0, 0])
            elif emotion == 'sadness':
                y_data.append([0, 0, 0, 1, 0, 0, 0])
            elif emotion == 'happiness':
                y_data.append([0, 0, 0, 0, 1, 0, 0])
            elif emotion == 'like':
                y_data.append([0, 0, 0, 0, 0, 1, 0])
            elif emotion == 'surprise':
                y_data.append([0, 0, 0, 0, 0, 0, 1])
            else:
                print('error')
    y_data = np.array(y_data)
    print(y_data.shape)
    return x_data, y_data


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



