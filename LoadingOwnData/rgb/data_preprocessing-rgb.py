import os
import pickle
import cv2

DATADIR="../../resources"

Categories = ["Dog", "Cat"]
IMG_SIZE = 100

image_data = []
image_labels = []

print("hello")

for category in Categories:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            image_data.append(new_array)
            image_labels.append(1 if category == "Dog" else 0)
        except Exception as e:
            pass


pickle_out = open("../../resources/rgb/X.pickle", "wb")
pickle.dump(image_data, pickle_out)
pickle_out.close()

pickle_out = open("../../resources/rgb/y.pickle", "wb")
pickle.dump(image_labels, pickle_out)
pickle_out.close()






