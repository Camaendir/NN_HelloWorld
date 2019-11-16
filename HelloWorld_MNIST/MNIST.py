import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_test))
print(len(y_test))

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


import random
zufall = random.randrange(0,9990)
print(zufall)
print(len(x_test))
predictions = model.predict(x_test)
for i in range(10):
    print("predict: " + str(np.argmax(predictions[zufall + i])) + " - actual: " + str(y_test[zufall + i]) + " - " + str(np.argmax(predictions[zufall + i]) == y_test[zufall + i]))
    plt.imshow(x_test[zufall + i], cmap=plt.cm.binary)
    plt.show()
    from time import sleep
    sleep(5)
