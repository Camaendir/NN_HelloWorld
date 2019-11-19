from tkinter import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

decoder = tf.keras.models.Sequential()
model = tf.keras.models.load_model("../autoencoder_mnist.model")
decoder.add(model.layers[4])
decoder.add(model.layers[5])
decoder.add(model.layers[6])


def insert():
    weights = []
    for scale in scales:
        weights.append((float(scale.get()))/100)
    weights = np.array(weights).reshape(-1, 32)
    decoded_img = decoder.predict(weights)
    n = len(decoded_img)
    for i in range(n):
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


m = Tk()
Button(m, text="insert", command=insert).pack()
frames = []
frame1 = Frame(m)
frame2 = Frame(m)
frame3 = Frame(m)
frame1.pack(padx=5, pady=10, side=LEFT)
frame2.pack(padx=5, pady=60, side=LEFT)
frame3.pack(padx=5, pady=110, side=LEFT)
frames.append(frame1)
frames.append(frame2)
frames.append(frame3)
scales = []
for i in range(32):
    print(min(int(i/10), 2))
    frame = frames[min(int(i/10), 2)]
    l = Label(frame, text=str(i + 1))
    l.pack()
    s = Scale(frame, from_=0, to=100, orient=HORIZONTAL)
    scales.append(s)
    s.pack()


while True:
    m.mainloop()