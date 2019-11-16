import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('catsndogs.model')

import pickle
IMG_SIZE = 100

pickle_in = open("../resources/X_training.pickle", "rb")
x_train = np.array(pickle.load(pickle_in)).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_in.close()

pickle_in = open("../resources/y_training.pickle", "rb")
y_train = np.uint8(pickle.load(pickle_in))
pickle_in.close()

pickle_in = open("../resources/X_test.pickle", "rb")
x_test = np.array(pickle.load(pickle_in)).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_in.close()

pickle_in = open("../resources/y_test.pickle", "rb")
y_test = np.uint8(pickle.load(pickle_in))
pickle_in.close()


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model.fit(x_train, y_train, epochs=50, verbose=True)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('catsndogs.model')