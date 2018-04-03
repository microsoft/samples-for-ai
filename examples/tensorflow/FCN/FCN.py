#MIT License
#Fully Convolutional Networks
#Based on VGG19


from __future__ import print_function
import tensorflow as tf
import scipy
import numpy as np
import os
from six.moves import cPickle as pickle
import glob
import random
import scipy.misc as misc
from scipy.io import loadmat
import time

FlAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("model_dir","model/","path to vgg model")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_float("keep_prob","1","keep_prob in dropout ")
tf.flags.DEFINE_integer("NUM_CLASS","151","numbers of class")
tf.flags.DEFINE_integer("IMAGE_SIZE","224","size of image")
tf.flags.DEFINE_integer("BATCH_SIZE","2","size of batch step")
tf.flags.DEFINE_integer("ITERATIONS","1000","the iterations of training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "learning rate")
tf.flags.DEFINE_string("data_dir", "datasets/", "path to dataset")
tf.flags.DEFINE_string("pickle","datasets.pickle","pickle for dataset")
tf.flags.DEFINE_string('mode', "train", "mode: train or visualize")
#url vgg19 model
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
#url dataset
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


#get model
def get_model_data(model_dir):
    filename = MODEL_URL.split("/")[-1]
    filepath = os.path.join(model_dir,filename)
    if not os.path.exists(filepath):
        print("there is no vgg19 model")
        return None
    model_data = loadmat(filepath)
    return model_data
model_data = get_model_data(FlAGS.model_dir)

#get data
def create_image_list(image_dir):
    if not os.path.exists(image_dir):
        print("image_dir not found")
        return None
    image_list = {}
    contents = ['training','validation']

    for content in contents:
        file_list = []
        image_list[content] = []
        file_list.extend(glob.glob(os.path.join(image_dir,"images",content,'*.'+'jpg')))

        for image_file in file_list:
            image_name = os.path.splitext(image_file.split("/")[-1])[0]
            annotation_file = os.path.join(image_dir,"annotations",content,image_name+'.png')
            record = {'image':image_file,'annotation':annotation_file,'filename':image_name}
            image_list[content].append(record)

        random.shuffle(image_list[content])
        print("%s files:%d"% (content,len(image_list[content])))

    return image_list


def create_pickle(data_dir):
    pickle_name = FlAGS.pickle
    pickle_path = os.path.join(data_dir,pickle_name)
    if not os.path.exists(pickle_path):
        dataset_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        image_list = create_image_list(os.path.join(data_dir,dataset_folder))
        with open(pickle_path,'wb') as pick:
            pickle.dump(image_list,pick,pickle.HIGHEST_PROTOCOL)
    # can devide into two part
    #return pick

    with open(pickle_path,'rb') as pick:
        result = pickle.load(pick)
        training_records = result['training']
        validation_records = result['validation']
        print(len(training_records))
        print(len(validation_records))
        del result
    return training_records,validation_records

#batch of dataset

def transform(filename,mode):
    image = misc.imread(filename)
    if len(image.shape)<3 and mode == "train" :
        image =  np.array([image for i in range(3)])
    resize_image = misc.imresize(image,[224,224],interp='nearest')
    image=resize_image
    return image

def get_dataset(train_record_list,mode):
    print(mode)
    file = train_record_list
    images = np.array([transform(filename['image'],mode) for filename in file])
    annotations = np.array([np.expand_dims(transform(filename['annotation'],"annotation"),axis=3) for filename in file])
    return images, annotations

def get_batch_train(images,annotations,offset,batch_size,size,epoch):
    start = offset
    offset += batch_size
    if offset>size:
        epoch += 1
        print("have done %d epochs.." % epoch)
        new = np.arange(size)
        np.random.shuffle(new)
        images = images[new]
        annotations = annotations[new]
        start = 0
        offset = batch_size
    end = offset
    return images[start:end],annotations[start:end],offset,epoch

def get_random_batch(images,annotations,batch_size,size):
    index = np.random.randint(0,size,batch_size).tolist()
    return images[index],annotations[index]





def vgg_model(model_data,image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    )
    net = {}
    current = image
    model_weights_bias = model_data['layers'][0]
    for i,name in enumerate(layers):
        if name[:4] == 'conv':
            weights,bias = model_weights_bias[i][0][0][0][0]
            weights = np.transpose(weights,(1,0,2,3))
            bias = bias.reshape(-1)
            weights = tf.get_variable(name=name + "_w",initializer=tf.constant_initializer(weights,dtype=tf.float32),shape=weights.shape)
            bias = tf.get_variable(name=name + "_b",initializer=tf.constant_initializer(bias,dtype=tf.float32),shape=bias.shape)
            conv = tf.nn.conv2d(current,weights,strides=[1,1,1,1],padding="SAME")
            current = tf.nn.bias_add(conv,bias)

        elif name[:4] == 'relu':
            current = tf.nn.relu(current,name=name)

        elif name[:4] == 'pool':
            current = tf.nn.avg_pool(current,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        net[name]=current

    return net


def fcn_model(image):

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean,axis=(0,1))
    image = image - mean_pixel # image process

    with tf.variable_scope("fcn_model"):
        image_net = vgg_model(model_data,image)
        conv5_4 = image_net["conv5_4"]
        pool5 = tf.nn.max_pool(conv5_4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        W_6 = tf.get_variable(name="conv6_w",initializer=tf.truncated_normal(shape=[7,7,512,4096],stddev=0.02))
        b_6 = tf.get_variable(name="conv6_b",initializer=tf.constant(0.0,shape=[4096]))

        conv6 = tf.nn.conv2d(pool5,W_6,strides=[1,1,1,1],padding="SAME")
        conv6_b = tf.nn.bias_add(conv6,b_6)
        relu6 = tf.nn.relu(conv6_b,name="relu6")
        drop6 = tf.nn.dropout(relu6,keep_prob=FlAGS.keep_prob)

        W_7 = tf.get_variable(name="conv7_w",initializer=tf.truncated_normal(shape=[1,1,4096,4096],stddev=0.02))
        b_7 = tf.get_variable(name="conv7_b",initializer=tf.constant(0.0,shape=[4096]))

        conv7 = tf.nn.conv2d(drop6, W_7, strides=[1,1,1,1], padding="SAME")
        conv7_b = tf.nn.bias_add(conv7, b_7)
        relu7 = tf.nn.relu(conv7_b, name="relu7")
        drop7 = tf.nn.dropout(relu7, keep_prob=FlAGS.keep_prob)

        W_8 = tf.get_variable(name="conv8_w",initializer=tf.truncated_normal(shape=[1,1,4096,FlAGS.NUM_CLASS],stddev=0.02))
        b_8 = tf.get_variable(name="conv8_b",initializer=tf.constant(0.0,shape=[FlAGS.NUM_CLASS]))

        conv8 = tf.nn.conv2d(drop7,W_8,strides=[1,1,1,1],padding="SAME")
        conv8_b = tf.nn.bias_add(conv8,b_8)

    # time to upscale

        deconv1_shape = image_net["pool4"].get_shape()
        deconv2_shape = image_net["pool3"].get_shape()
        image_shape = tf.shape(image)
        deconv3_shape = tf.stack([image_shape[0],image_shape[1],image_shape[2],FlAGS.NUM_CLASS])

        TW_1 = tf.get_variable(name="deconv1_w",initializer=tf.truncated_normal(shape=[4,4,deconv1_shape[3].value,FlAGS.NUM_CLASS],stddev=0.02))
        Tb_1 = tf.get_variable(name="deconv1_b",initializer=tf.constant(0.0,shape=[deconv1_shape[3].value]))
        deconv1 = tf.nn.conv2d_transpose(conv8_b,TW_1,output_shape=tf.shape(image_net["pool4"]),strides=[1,2,2,1],padding="SAME")
        deconv1_b = tf.nn.bias_add(deconv1,Tb_1)
        fuse_1 = tf.add(deconv1_b,image_net["pool4"],name="fuse_1")

        TW_2 = tf.get_variable(name="deconv2_w",initializer=tf.truncated_normal(shape=[4,4,deconv2_shape[3].value,deconv1_shape[3].value],stddev=0.02))
        Tb_2 = tf.get_variable(name="deconv2_b", initializer=tf.constant(0.0,shape=[deconv2_shape[3].value]))
        deconv2 = tf.nn.conv2d_transpose(fuse_1, TW_2, output_shape=tf.shape(image_net["pool3"]), strides=[1, 2, 2, 1],padding="SAME")
        deconv2_b = tf.nn.bias_add(deconv2, Tb_2)
        fuse_2 = tf.add(deconv2_b,image_net["pool3"],name="fuse_2")

        TW_3 = tf.get_variable(name="deconv3_w",initializer=tf.truncated_normal(shape=[16,16,FlAGS.NUM_CLASS,deconv2_shape[3].value],stddev=0.02))
        Tb_3 = tf.get_variable(name="deconv3_b",initializer=tf.constant(0.0,shape=[FlAGS.NUM_CLASS]))
        deconv3 = tf.nn.conv2d_transpose(fuse_2,TW_3,output_shape=deconv3_shape,strides=[1,8,8,1],padding="SAME")
        deconv3_b = tf.nn.bias_add(deconv3,Tb_3)

        annotation_pred = tf.argmax(deconv3_b,dimension=3,name="prediction")
        annotation_pred = tf.expand_dims(annotation_pred, dim=3)

        return annotation_pred,deconv3_b


def train(loss,var_list):
    optimizer = tf.train.AdamOptimizer(FlAGS.learning_rate)
    grads = optimizer.compute_gradients(loss,var_list=var_list)
    return optimizer.apply_gradients(grads)

def main(argv=None):
    starttime = time.time()
    image = tf.placeholder(tf.float32,shape=[None,FlAGS.IMAGE_SIZE,FlAGS.IMAGE_SIZE,3],name="input_image")
    annotation = tf.placeholder(tf.int32,shape=[None,FlAGS.IMAGE_SIZE,FlAGS.IMAGE_SIZE,1],name="input_annotation") ## tips

    pred_annotation,logits = fcn_model(image)

    squeeze_annotation = tf.squeeze(annotation,squeeze_dims=[3])
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=squeeze_annotation,name="entropy")))
    print(" train_op ")
    trainable_var = tf.trainable_variables()
    train_op = train(loss,trainable_var)

    print(time.time()-starttime)

    print("ready for records")
    # ready for records
    train_records, validation_records = create_pickle(FlAGS.data_dir)
    print("ready for datasets")
    print(time.time()-starttime)
    train_validation_dataset = "train_validation_dataset.npz"
    npzfile_path = os.path.join(FlAGS.data_dir,train_validation_dataset)
    if not os.path.exists(npzfile_path):
        validation_image_datasets,validation_annotation_datasets = get_dataset(validation_records,"validation")
        print(validation_image_datasets.shape)
        print(validation_annotation_datasets.shape)
        train_image_datasets,train_annotation_datasets = get_dataset(train_records,"train")
        print(train_image_datasets.shape)
        print(train_annotation_datasets.shape)
        np.savez(npzfile_path,train_image_datasets,train_annotation_datasets,validation_image_datasets,validation_annotation_datasets)
    else:
        print("datasets are loading")
        datasets = np.load(npzfile_path)
        train_image_datasets = datasets["arr_0"]
        train_annotation_datasets = datasets["arr_1"]
        validation_image_datasets = datasets["arr_2"]
        validation_annotation_datasets = datasets["arr_3"]
    #save npz
    print(time.time()-starttime)

    print("ready for saver")
    #ready for saver
    saver = tf.train.Saver()

    sess = tf.Session()

    ckpt_file = tf.train.get_checkpoint_state(FlAGS.logs_dir)

    sess.run(tf.global_variables_initializer())

    #restore ckpt
    if FlAGS.mode == "visualize" and ckpt_file.model_checkpoint_path:
        print("model restored")
        saver.restore(sess, ckpt_file.model_checkpoint_path)

    sum_train_size = train_image_datasets.shape[0]
    sum_validation_size = validation_image_datasets.shape[0]
    if FlAGS.mode == "train":
        offset = 0
        epoch = 0
        for i in range(FlAGS.ITERATIONS):
            print(i)
            train_batch_image, train_batch_annotation,offset,epoch = get_batch_train(train_image_datasets,
                                                                                     train_annotation_datasets,
                                                                                     offset,FlAGS.BATCH_SIZE,
                                                                                     sum_train_size,epoch)
            sess.run(train_op,feed_dict={image:train_batch_image,annotation:train_batch_annotation})
            print(offset)
            if i%10 == 0:
                train_loss = sess.run(loss,feed_dict={image:train_batch_image,annotation:train_batch_annotation})
                print("the loss of iteration: %d is %g"% (i,train_loss))

            if i % 500 == 0:
                validation_batch_image, validation_batch_annotation, offset, epoch = get_batch_train(validation_image_datasets,
                                                                                           validation_annotation_datasets,
                                                                                           offset, FlAGS.BATCH_SIZE,
                                                                                           sum_validation_size, epoch)
                valid_loss = sess.run(loss, feed_dict={image: validation_batch_image, annotation: validation_batch_annotation})
                print("iteration %d --loss of validation : %g" % (i, valid_loss))
                saver.save(sess, FlAGS.logs_dir + "model.ckpt" ,i)

    elif FlAGS.mode == "visualize":
        print(" visualize ")
        validation_batch_image,validation_batch_annotation = get_random_batch(validation_image_datasets,
                                                                              validation_annotation_datasets,
                                                                              FlAGS.BATCH_SIZE,
                                                                              sum_validation_size)
        pred = sess.run(pred_annotation, feed_dict={image: validation_batch_image, annotation: validation_batch_annotation})
        valid_annotations = np.squeeze(validation_batch_annotation, axis=3)
        pred = np.squeeze(pred, axis=3)
        for i in range(FlAGS.BATCH_SIZE):
            print("save image..")
            misc.imsave(os.path.join(FlAGS.logs_dir,"pred_"+str(i)+".png"),pred[i].astype(np.uint8))
            misc.imsave(os.path.join(FlAGS.logs_dir,"val_"+str(i)+".png"),valid_annotations[i].astype(np.uint8))
            misc.imsave(os.path.join(FlAGS.logs_dir,"image_"+str(i)+".png"),validation_batch_image[i].astype(np.uint8))


if __name__ == "__main__":
    tf.app.run()