
# coding: utf-8



import glob
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import keras
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Add, LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Merge
from keras.layers import concatenate
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model
from IPython.display import clear_output



#The dataset can be obtained in  http://lixirong.net/datasets/flickr8kcn

token = '/datasets/Flicker8k_text/Flickr8k.token.txt'
captions = open(token, 'r').read().strip().split('\n')
d = {}


for i, row in enumerate(captions):
    row = row.split('\t')
    row[0] = row[0][:len(row[0])-2]
    if row[0] in d:
        d[row[0]].append(row[1])
    else:
        d[row[0]] = [row[1]]



images = '/datasets/Flicker8k_Dataset/'
img = glob.glob(images+'*.jpg')

train_images_file = '/datasets/Flicker8k_text/Flickr_8k.trainImages.txt'
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

def split_data(l):
    temp = []
    for i in l:
        temp.append(images+i)
    return temp

train_img = split_data(train_images)


val_images_file = '/datasets/Flicker8k_text/Flickr_8k.devImages.txt'
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))

# Getting the validation images from all the images
val_img = split_data(val_images)
len(val_img)
test_images_file = '/datasets/Flicker8k_text/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Getting the testing images from all the images
test_img = split_data(test_images)




len(test_img)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x



def preprocess(path):
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x




from keras.models import Model
model = InceptionV3(weights='imagenet')

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)




import os

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc
encoding_train = {}
#encoding the images and saving in pickle
if os.path.exists('encoded_images_inceptionV3.p') != True:
    for img in tqdm(train_img):
        encoding_train[img[len(images):]] = encode(img)
    with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_train, encoded_pickle)
else:
    encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))


encoding_test = {}

if os.path.exists('encoded_images_test_inceptionV3.p') != True:
    for img in tqdm(test_img):
        encoding_test[img[len(images):]] = encode(img)
    with open("encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_test, encoded_pickle)
else:
    encoding_test = pickle.load(open('encoded_images_test_inceptionV3.p', 'rb'))



encoding_val = {}

if os.path.exists('encoded_images_val_inceptionV3.p') != True:
    for img in tqdm(val_img):
        encoding_val[img[len(images):]] = encode(img)
    with open("encoded_images_val_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_val, encoded_pickle)
else:
    encoding_val =  pickle.load(open('encoded_images_val_inceptionV3.p', 'rb'))



train_d = {}
for i in train_img:
    if i[len(images):] in d:
        train_d[i] = d[i[len(images):]]


val_d = {}
for i in val_img:
    if i[len(images):] in d:
        val_d[i] = d[i[len(images):]]

test_d = {}
for i in test_img:
    if i[len(images):] in d:
        test_d[i] = d[i[len(images):]]

captions = []
for key, val in train_d.items():
    for i in val:
        captions.append('<start> ' + i + ' <end>')



captions_all = []
all_img = []
for i in img:
    all_img.append(i)

all_d = {}  

for i in all_img:
    if i[len(images):] in d:
        all_d[i] = d[i[len(images):]]

captions_all = []
for key, val in all_d.items():
    for i in val:
        captions_all.append('<start> ' + i + ' <end>')


words = [i.split() for i in captions_all]

unique = []
for i in words:
    unique.extend(i)
    unique = list(set(unique))



word2idx = {val:index for index, val in enumerate(unique)}
idx2word = {index:val for index, val in enumerate(unique)}


vocab_size = len(unique)



f = open('flickr8k_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")

# creating table in file <image_id>\t<caption>
for key, val in train_d.items():
    for i in val:
        f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")

f.close()

df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')

c = [i for i in df['captions']]

imgs = [i for i in df['image_id']]

samples_per_epoch = 0

for ca in captions:
    samples_per_epoch += len(ca.split()) - 1




max_len = 40




captions_val = []
for key, val in val_d.items():
    for i in val:
        captions_val.append('<start> ' + i + ' <end>')

f = open('flickr8k_validation_dataset.txt', 'w')
f.write("image_id\tcaptions\n")

# creating table in file <image_id>\t<caption>


# creating table in file <image_id>\t<caption>
for key, val in val_d.items():
    for i in val:
        f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")

f.close()

df = pd.read_csv('flickr8k_validation_dataset.txt', delimiter='\t')

c = [i for i in df['captions']]
imgs = [i for i in df['image_id']]

num_val = 0
for ca in captions_val:
    num_val += len(ca.split()) - 1



def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []

        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1

                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)

                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)

                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        # x=[[images, partial_caps], next_words]
                        # print(x.shape[0], x.shape[1])
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0
                        
def data_generator_val(batch_size = 512):
        partial_caps = []
        next_words = []
        images = []

        df = pd.read_csv('flickr8k_validation_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_val[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1

                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)

                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)

                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        # x=[[images, partial_caps], next_words]
                        # print(x.shape[0], x.shape[1])
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0



file = open("accuracy.txt", 'w')
file1 = open("loss.txt", 'w')
file2 = open("accuracy_val.txt", 'w')
file3 = open("loss_val.txt", 'w')

# recording acc and loss after each batch
class PlotLosses(Callback):

    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
     
        #if self.batch % self.N == 0:
        #    name = './weights_after_batches/weights%08d.h5' % batch
        #    self.model.save_weights(name)
        self.batch += 1

    def on_epoch_end(self, epoch, logs={}):
        file.write("%s\n" %logs.get('acc') )
        file1.write("%s\n" %logs.get('loss'))
        file2.write("%s\n" %logs.get('val_acc') )
        file3.write("%s\n" %logs.get('val_loss'))   

 

embedding_size = 300

image_model = Sequential([
       Dense(embedding_size, input_shape=(2048,), activation='relu'),
      RepeatVector(max_len)
      ])

caption_model = Sequential([
          Embedding(vocab_size, embedding_size, input_length=max_len),
          LSTM(256, return_sequences=True),
           TimeDistributed(Dense(300))
                    ])

# merging the models

final_model = Sequential([
                        Merge([image_model, caption_model], mode='concat', concat_axis=1),
                        Bidirectional(LSTM(256, return_sequences=False)),
                        Dense(vocab_size),
                        Activation('softmax')
                    ])

final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

final_model.summary()
final_model.fit_generator(data_generator(batch_size=512), steps_per_epoch=748,  validation_data=data_generator(batch_size = 512),validation_steps = 125, workers=4, use_multiprocessing=True,callbacks=[PlotLosses(final_model, 10)],
                                         nb_epoch=57)
# save the best weight after training

file.close()
file1.close()
file2.close()
file3.close()

model.save("best_weight.hdf5", overwrite= True)


def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > max_len:
            break


        return ' '.join(start_word[1:-1])


def beam_search_predictions(image, beam_index=3):
    start = [word2idx["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])

            word_preds = np.argsort(preds[0])[-beam_index:]

            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top word
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption
