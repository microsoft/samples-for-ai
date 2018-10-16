"""

Example of an autoencoder made in tensorflow.
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

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


# Define some important stuff
img_size = 28      # 28 because MNIST is composed of 28 x 28 images
img_channel = 1    #  1 because MNIST is grayscale

batch_size = 128
train_steps = 20000    # This number is an overshoot

image_save_step = 1000

# Number of neurons in the first layer.
img_total = img_size * img_size * img_channel

layer_size1 = round(img_total * 0.5)

# Number of neurons in the second layer.
layer_size2 = round(img_total * 0.3)

# Number of neurons in the middle (third) layer (the compressed state).
layer_size3 = round(img_total * 0.10)


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', help='Directory with the input data.')
parser.add_argument('--output', type=str, default='output', help='Directory where generated images will be saved.')

FLAGS, unparsed = parser.parse_known_args()
INPUT_DIR = FLAGS.input
OUTPUT_DIR = FLAGS.output


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# To get images
def get_images():

    def get_data():

        if INPUT_DIR:
            data = []

            for file in os.listdir(INPUT_DIR):
                img = Image.open(INPUT_DIR + "/" + file)
                img = img.resize((img_size, img_size), Image.ANTIALIAS)  # Resize to target size.
                img_data = np.array(img)
                data.append(img_data)

            return np.array(data)
        else:  # Should use MNIST
            print("No input data was given, script will use MNIST by default.")
            (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
            return np.expand_dims(x_train, 3)

    # Since the values of the data is in [0, 255],
    # we rescale the values of the array,
    # so that they are in [-1, 1]
    return (get_data() - 127.5) / 127.5


# The encoder will receive an image
# and compress it into a small layer, returning it.
def get_encoder(x):

    with tf.name_scope("encoder"):

        flatten1 = tf.layers.flatten(x, name="flatten_layer")

        dense1 = tf.layers.dense(flatten1, layer_size1, name="first_encoder_layer")
        relu1 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(relu1, layer_size2, name="second_encoder_layer")
        relu2 = tf.nn.relu(dense2)

        dense3 = tf.layers.dense(relu2, layer_size3, name="compressed_encoder_layer")  # compressed state
        relu3 = tf.nn.relu(dense3)

        return relu3


# The decoder will receive the encoded state
# and decode into the original.
def get_decoder(x):

    with tf.name_scope("decoder"):

        dense1 = tf.layers.dense(x, layer_size2, name="first_decoder_layer")
        relu1 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(relu1, layer_size1, name="second_decoder_layer")
        relu2 = tf.nn.relu(dense2)

        dense3 = tf.layers.dense(relu2, img_size * img_size * img_channel, name="third_decoder_layer")

        reshape = tf.reshape(dense3, [-1, img_size, img_size, img_channel], name="image_reshape")
        tanh = tf.nn.tanh(reshape)  # Tanh so it's limited to [-1, +1], same of input data.

        return tanh, reshape


# Define the placeholder
img_ph = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channel], name="images_ph")

# Define the network that will be trained
# The encoder will be built with the img placeholder
encoder = get_encoder(img_ph)

# The decoder will receive the encoded data.
decoder_out, decoder_logits = get_decoder(encoder)

# Create a tensor that will be saved to TensorBoard
# Concat the input and output images, allowing us to see the difference.
image_save_op = tf.concat([img_ph, decoder_out], 2, name="image_saver_op")

# Now, we need to define the loss
with tf.name_scope("loss"):
    # A simple mean squared error
    # Since what we want is the network to rebuild the image, the target is img_ph, the input.
    # Using logits so that the backprop is more efficient.
    loss_op = tf.losses.mean_squared_error(img_ph, decoder_logits)

# The optimizer used will be Adam, change the learn rate if needed.
with tf.name_scope("optimizer"):
    opt_op = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss_op)

# We then load the data
x_train = get_images()

# Op to initializer all variables
init = tf.global_variables_initializer()

# Session time!
with tf.Session() as sess:

    # Run the init
    sess.run(init)

    # TensorBoard Stuff

    # FileWriter to save the log
    writer = tf.summary.FileWriter("logs/", graph=sess.graph)

    # A summary to save generated images
    image_saver_summary = tf.summary.image("Decoded Images", image_save_op, max_outputs=9)

    # A summary to save the loss
    merged_sum = tf.summary.merge([tf.summary.scalar("Loss", loss_op)])

    # End of TensorBoard Stuff

    # To memorize the start time
    train_start_time = time.time()

    # Defines the main train loop
    for step in range(train_steps):

        # Random number to select the images
        random_number = np.random.randint(0, len(x_train) - batch_size)

        # Slice the train data to create a batch
        img_batch = x_train[random_number: random_number + batch_size]

        # Train the network
        _, current_loss, summary = sess.run([opt_op, loss_op, merged_sum],
                                            feed_dict={img_ph: img_batch})

        # Saves TensorBoard stuff
        writer.add_summary(summary, step)

        # Print info about the step
        print("Step {0}, LOSS {1:.15}".format(step, current_loss))

        # Occasionally saves images to TensorBoard
        if step % image_save_step == 0 or step == train_steps - 1:
            print("SAVING IMAGE...")

            # Random number to select the images
            random_number = np.random.randint(0, len(x_train) - batch_size)

            # Slice the train data to create a batch
            img_batch = x_train[random_number: random_number + batch_size]

            # Run the summary op.
            images, save_sum = sess.run([image_save_op, image_saver_summary], feed_dict={img_ph: img_batch})

            # Saves sample images to TensorBoard
            writer.add_summary(save_sum, step)

            # Save with matplotlib
            # Rescale the images to [0, 1] (needed for matplotlib)
            images = (images * 0.5) + 0.5

            # Because grayscale
            images = np.squeeze(images)  # Comment this line if your dataset is not GRAYSCALE

            # Number of images to save
            r, c = 4, 4

            # Creates subplots
            fig, axs = plt.subplots(r, c)

            # Loop to save the images
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(images[cnt], cmap="gray")  # Remove the 'cmap' if your dataset is not GRAYSCALE
                    axs[i, j].axis('off')
                    cnt += 1

            # Set title
            fig.suptitle("Sample after {0} training steps".format(step), fontsize=14)
            # Saves
            fig.savefig("{0}/generated_{1}.png".format(OUTPUT_DIR, step))
            # Close the plt
            plt.close()


# Print
spent_time = time.time() - train_start_time
print("Trained done in {0} seconds, average of {1} steps/second".format(spent_time, train_steps / spent_time))
