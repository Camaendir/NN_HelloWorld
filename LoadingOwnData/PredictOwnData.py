import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time

current_milli_time = lambda: int(round(time.time() * 1000))

isDirectory = True
PATH = "OwnTestData"
isGraycale = False

model_name = "catsndogs.model" if isGraycale else "rgb/catsndogs_rgb.model"
model = tf.keras.models.load_model(model_name)

if not isDirectory:

    if isGraycale:
        img_full = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    else:
        img_full = cv2.imread(PATH, cv2.IMREAD_COLOR)
    img_small = cv2.resize(img_full, (100, 100))
    layer = 1 if isGraycale else 3
    if isGraycale:
        np_img = np.array(img_small).reshape(-1, 100, 100, 1)
    else:
        np_img = np.array(img_small).reshape(-1, 100, 100, 3)
    normalized = tf.keras.utils.normalize(np_img, axis=layer)
    prediction = model.predict(normalized)
    type = "Dog" if np.argmax(prediction) == 1 else "Cat"
    print("Cat - " + str(prediction) + " - Dog")
    plt.text(int(img_full.shape[1] / 2), int(img_full.shape[0] + 100), type)
    plt.margins(0)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR), cmap=plt.cm.binary)
    plt.show()
else:
    all_img = []
    all_img_full = []
    layers = 1 if isGraycale else 3
    before = current_milli_time()
    for file in os.listdir(PATH)[:25]:
        if isGraycale:
            full_img = cv2.imread(os.path.join(PATH, file), cv2.IMREAD_GRAYSCALE)
        else:
            full_img = cv2.imread(os.path.join(PATH, file))
        all_img_full.append(full_img)
        resize_img = cv2.resize(full_img, (100, 100))
        all_img.append(resize_img)
    np_array = np.array(all_img).reshape(-1 , 100, 100, layers)
    normalized = tf.keras.utils.normalize(np_array, axis=1)
    mid = current_milli_time()
    predictions = model.predict(normalized)
    after = current_milli_time()
    print("Read and Preprocess Image: " + str(mid - before))
    print("Evaluate all Images: " + str(after - mid))
    print("Everyting: " + str(after - before))
    for i in range(len(predictions)):
        type = "Dog" if np.argmax(predictions[i]) == 1 else "Cat"
        print("Cat - " + str(predictions[i]) + " - Dog")
        plt.imshow(cv2.cvtColor(all_img_full[i], cv2.COLOR_RGB2BGR))
        plt.axis("off")
        plt.text(int(all_img_full[i].shape[1] / 2), int(all_img_full[i].shape[0] + 100), type)
        plt.show()
        sleep(5)