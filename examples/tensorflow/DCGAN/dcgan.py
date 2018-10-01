"""

Example of a deep convolutional generative adversarial network.
This example uses the layer api in tensorflow, in order to keep things simpler and easier, but powerfull.

How to use:
Put training data (images) inside 'input' folder (create one if there is none) or pass folder name with --input_dir.
Tweak the values, if necessary.
PROFIT

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',type=str, default='input', help='Directory with the input data.')
parser.add_argument('--output_dir',type=str, default='output', help='Directory where generated images will be saved.')

FLAGS, unparsed = parser.parse_known_args()
INPUT_DIR = FLAGS.input_dir
OUTPUT_DIR = FLAGS.output_dir

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define some important stuff
img_size = 128
img_channel = 3

noise_dim = 100

batch_size = 64
train_steps = 1

image_save_step = 500

# Kernel size
k_s = 5


# To get images
def get_images():

    files = ["{}/{}".format(INPUT_DIR, file) for file in os.listdir(INPUT_DIR)]

    data = []

    for file in files:

        img = Image.open(file)
        img = img.resize((img_size, img_size), Image.ANTIALIAS)  # Resize to target size.

        img_data = np.array(img)

        data.append(img_data)

    return np.array(data)


# Image -> classification (real or fake)
def get_dis(x, reuse=False, train=True):

    with tf.variable_scope("dis_", reuse=reuse):

        conv1 = tf.layers.conv2d(x, 64, k_s, strides=[4, 4], padding="SAME", trainable=train)
        norm1 = tf.layers.batch_normalization(conv1, trainable=train)
        lrelu1 = tf.nn.leaky_relu(norm1)

        conv2 = tf.layers.conv2d(lrelu1, 128, k_s, strides=[2, 2], padding="SAME", trainable=train)
        norm2 = tf.layers.batch_normalization(conv2, trainable=train)
        lrelu2 = tf.nn.leaky_relu(norm2)

        flatten1 = tf.layers.flatten(lrelu2)
        dense1 = tf.layers.dense(flatten1, units=1, trainable=train)

        out = tf.nn.sigmoid(dense1)

        return out, dense1


# Noise -> Fake Image
def get_gen(x, reuse=False, train=True):

    with tf.variable_scope("gen_", reuse=reuse):

        # Use the fourth part of the img_size to correctly scale
        quarter = img_size // 4

        # Dense to amplify noise, reshape to turn it into Conv2d input
        dense1 = tf.layers.dense(x, units=quarter*quarter*128)
        reshape1 = tf.reshape(dense1, shape=[-1, quarter, quarter, 128])

        conv1 = tf.layers.conv2d_transpose(reshape1, 128, k_s, strides=[2, 2], padding="SAME", trainable=train)
        norm1 = tf.layers.batch_normalization(conv1, trainable=train)
        relu1 = tf.nn.relu(norm1)

        conv2 = tf.layers.conv2d_transpose(relu1, 64, k_s, strides=[2, 2], padding="SAME", trainable=train)
        norm2 = tf.layers.batch_normalization(conv2, trainable=train)
        relu2 = tf.nn.relu(norm2)

        conv3 = tf.layers.conv2d_transpose(relu2, img_channel, k_s, strides=[1, 1], padding="SAME", trainable=train)

        # Tanh to keep the values in [-1, 1]
        out = tf.nn.tanh(conv3)

        return out


# Define the placeholders
z_ph = tf.placeholder(tf.float32, shape=[None, noise_dim], name="noise_ph")
img_ph = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channel], name="images_ph")

# Define the generator that will be trained
gen = get_gen(z_ph)

# Define the discriminators, one receives real image place holders and the other receives the generator,
# this way creating a combined network (noise -> generator -> fake image -> discriminator -> classification
d_real_out, d_real_logits = get_dis(img_ph)
d_fake_out, d_fake_logits = get_dis(gen, reuse=True)

# Other generator, with reuse and that will not be trained,
# will only be used to visualize images
gen_off = get_gen(z_ph, reuse=True, train=False)

# Now, we need to define the losses
with tf.name_scope("losses"):

    # The discriminator total loss will be the sum
    # of the loss of the two discriminators, using the logits
    with tf.name_scope("dis_losses"):
        # The real loss will receive a tensor made of ones, since the images are from the dataset
        d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))

        # The fake loss will receive a tensor made of zeros, since the generated images are not from the dataset
        d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))

        # So, the total loss will be the sum
        d_loss = d_real_loss + d_fake_loss

    with tf.name_scope("gen_loss"):
        # The generator loss is almost the same as the ´d_fake_loss´, but receives a tensor made of
        # ones, since we want the generator to fool the discriminator
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))

# Making use of the variable name scope defined in the get_gen and get_dis,
# we are able to split the trainable variables into
# the ones that belongs to the gen and the ones that belongs to the dis
train_vars = tf.trainable_variables()
gen_vars = [v for v in train_vars if v.name.startswith("gen_")]
dis_vars = [v for v in train_vars if v.name.startswith("dis_")]

# Using the var_lists, we create the Optimizers:
# One that trains the discriminator vars and minimizes the d_loss
# One that trains the generator vars and minimizes the g_loss
with tf.name_scope("Optimizers"):
    d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=dis_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(g_loss, var_list=gen_vars)

# We then load the data
x_train = get_images()

# Using only a part of the dataset is optional
#x_train = x_train[:1000]

# Since the values of the data is in [0, 255],
# we rescale the values of the data,
# so that they are in [-1, 1],
# the same as our generator output activation (tanh)
x_train = (x_train - 127.5) / 127.5

# Op to initializer all variables
init = tf.global_variables_initializer()

# Session time!
with tf.Session() as sess:

    # Run the init
    sess.run(init)

    # Tensorboard Stuff

    # FileWriter to save the log
    writer = tf.summary.FileWriter("logs/", graph=sess.graph)

    # A summary to save generated images
    image_saver_summary = tf.summary.image("Generated Image", gen_off, max_outputs=9)

    # A summary to save info about the discriminator
    merged_dis_sum = tf.summary.merge([tf.summary.scalar("dis_real_loss", d_real_loss),
                                      tf.summary.scalar("dis_fake_loss", d_fake_loss),
                                      tf.summary.scalar("dis_loss", d_loss)])

    # A summary to save info about the generator (no need for merge here)
    merged_gen_sum = tf.summary.merge([tf.summary.scalar("gen_loss", g_loss)])

    # End of Tensorboard Stuff

    # To memorize the start time
    train_start_time = time.time()

    # Defines the main train loop
    for step in range(train_steps):

        # The noise will be from a uniform distribution [-1, 1]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])

        # Random number to select the images
        random_number = np.random.randint(0, len(x_train) - batch_size)
        # Some real images
        imgs_batch = x_train[random_number: random_number + batch_size]

        # Train the discriminator (only once)
        _, current_d_loss, d_sum = sess.run([d_opt, d_loss, merged_dis_sum],
                                            feed_dict={img_ph: imgs_batch, z_ph: noise})

        # Trains the generator (two times)
        _, _, current_g_loss, g_sum = sess.run([g_opt, g_opt, g_loss, merged_gen_sum],
                                               feed_dict={z_ph: noise})

        # Saves TensorBoard stuff
        writer.add_summary(d_sum, step)
        writer.add_summary(g_sum, step)

        # Print info about the step
        print("Step {0}, DLOSS {1:.15}, GLOSS {2:.15}".format(step, current_d_loss, current_g_loss))

        # Occasionally saves a image
        if step % image_save_step == 0 or step == train_steps - 1:
            print("SAVING IMAGE...")

            # Number of images to generate = r * c
            r, c = 4, 4
            # Again, noise from a uniform distribution [-1, 1]
            noise = np.random.uniform(-1.0, 1.0, size=[r*c, noise_dim])

            # Get images from the ´gen_off´
            generated_imgs, save_img_sum = sess.run([gen_off, image_saver_summary],
                                                    feed_dict={z_ph: noise})

            # Saves sample images to TensorBoard
            writer.add_summary(save_img_sum, step)

            # Rescale the images to [0, 1] (needed for matplotlib)
            generated_imgs = (generated_imgs * 0.5) + 0.5

            # Creates subplots
            fig, axs = plt.subplots(r, c)

            # Loop to save the images
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(generated_imgs[cnt])
                    axs[i, j].axis('off')
                    cnt += 1

            # Saves
            fig.savefig(f"{OUTPUT_DIR}/generated_{step}.png")
            # Close the plt
            plt.close()

# Print
spent_time = time.time() - train_start_time
print(f"Trained done in {spent_time} seconds, average of {train_steps / spent_time} steps/second")
