import os
import random

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

encoding_dim = 32

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(500, activation="relu"))
# model.add(tf.keras.layers.Dense(200, activation="relu"))
# model.add(tf.keras.layers.Dense(encoding_dim, activation="sigmoid"))
# model.add(tf.keras.layers.Dense(200, activation="relu"))
# model.add(tf.keras.layers.Dense(500, activation="relu"))
# model.add(tf.keras.layers.Dense(784, activation="sigmoid"))
#
#model.compile(optimizer="adam", loss="binary_crossentropy")

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# model.fit(x_train, x_train, epochs=20, validation_data=(x_test, x_test))
#
# model.save('autoencoder_mnist.model')

model = tf.keras.models.load_model('autoencoder_mnist.model')

encoder = tf.keras.models.Sequential()
decoder = tf.keras.models.Sequential()

encoder.add(model.layers[0])
encoder.add(model.layers[1])
encoder.add(model.layers[2])
encoder.add(model.layers[3])

#decoder.add(model.layers[3])
decoder.add(model.layers[4])
decoder.add(model.layers[5])
decoder.add(model.layers[6])

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
offset = int(random.random() * 1000)
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i + offset].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i + offset].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
