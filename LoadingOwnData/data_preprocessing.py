import numpy as mp
import matplotlib.pyplot as plt
import os
import cv2

DATADIR="../resources"

Categories = ["Dog", "Cat"]
IMG_SIZE = 100

image_data = []
image_labels = []

print("hello")

for category in Categories:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            image_data.append(new_array)
            image_labels.append(1 if category == "Dog" else 0)
        except Exception as e:
            pass
print(len(image_data))
import pickle

pickle_out = open("../resources/X.pickle", "wb")
pickle.dump(image_data, pickle_out)
pickle_out.close()

pickle_out = open("../resources/y.pickle", "wb")
pickle.dump(image_labels, pickle_out)
pickle_out.close()






