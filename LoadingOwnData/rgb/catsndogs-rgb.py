import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

IMG_SIZE = 100

pickle_in = open("../../resources/rgb/X_training.pickle", "rb")
x_train = np.array(pickle.load(pickle_in)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
pickle_in.close()

pickle_in = open("../../resources/rgb/y_training.pickle", "rb")
y_train = np.uint8(pickle.load(pickle_in))
pickle_in.close()

pickle_in = open("../../resources/rgb/X_test.pickle", "rb")
x_test = np.array(pickle.load(pickle_in)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
pickle_in.close()

pickle_in = open("../../resources/rgb/y_test.pickle", "rb")
y_test = np.uint8(pickle.load(pickle_in))
pickle_in.close()


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(50, 50, 3)))
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(25, 25, 3)))
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(126, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('catsndogs_rgb.model')


import random
zufall = random.randrange(0,900)
predictions = model.predict(x_test)
for i in range(10):
    print("predict: " + str(np.argmax(predictions[zufall + i])) + " - actual: " + str(y_test[zufall + i]) + " - " + str(np.argmax(predictions[zufall + i]) == y_test[zufall + i]))
