import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

IMAGEDIR = "../resources/Positioning"

keras = tf.keras

model = keras.models.load_model("../models/catsndogs_rgb.model")
images = []
for file in os.listdir(IMAGEDIR):
    images.append(cv2.imread(os.path.join(IMAGEDIR, file), cv2.IMREAD_COLOR))

parts = 5
for image in images:
    height, width, bla = image.shape
    h_step = height / parts
    w_step = width / parts
    max_fitness = (0,0)
    for X_Start in range(parts):
        print("X" + str(X_Start))
        for X_length in range(parts - X_Start - 1):
            for Y_Start in range(parts):
                #print("Y" + str(Y_Start))
                for Y_length in range(parts - Y_Start - 1):
                    crop_image = image[int(h_step * Y_Start):int(h_step * Y_Start + h_step * (Y_length + 1)), int(w_step * X_Start):int(w_step * X_Start + (X_length + 1))]
                    resize_image = cv2.resize(crop_image, (100, 100))
                    np_array = np.array(resize_image).reshape(-1, 100, 100, 3)
                    normalized = keras.utils.normalize(np_array, axis=1)
                    prediction = model.predict(normalized)
                    fitness = prediction[0][1] * (1/(Y_length + 1)*(X_length + 1)*0.005) * (1/(abs(Y_length - X_length) + 1)*10)
                    if fitness > max_fitness[0]:
                        max_percentage = (fitness, X_Start, X_length + 1, Y_Start, Y_length + 1)
    plt.imshow(image)
    plt.axis("off")
    ax = plt.gca()
    rect = patches.Rectangle((max_percentage[1]*w_step, max_percentage[3]*h_step), max_percentage[2]*w_step, max_percentage[4]*h_step, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()