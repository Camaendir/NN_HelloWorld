import pickle
pickle_in = open("../../resources/rgb/X.pickle", "rb")
X_full_data = pickle.load(pickle_in)


pickle_in = open("../../resources/rgb/y.pickle", "rb")
y_full_data = pickle.load(pickle_in)

import random

combined = list(zip(X_full_data, y_full_data))
random.shuffle(combined)
X_full_data[:], y_full_data[:] = zip(*combined)

length = len(X_full_data)
X_training_data = X_full_data[:int(length * 6 / 7)]
y_training_data = y_full_data[:int(length * 6 / 7)]

X_test_data = X_full_data[int(length * 6 / 7):]
y_test_data = y_full_data[int(length * 6 / 7):]

pickle_out = open("../../resources/rgb/X_training.pickle", "wb")
pickle.dump(X_training_data, pickle_out)
pickle_out.close()

pickle_out = open("../../resources/rgb/y_training.pickle", "wb")
pickle.dump(y_training_data, pickle_out)
pickle_out.close()

pickle_out = open("../../resources/rgb/X_test.pickle", "wb")
pickle.dump(X_test_data, pickle_out)
pickle_out.close()

pickle_out = open("../../resources/rgb/y_test.pickle", "wb")
pickle.dump(y_test_data, pickle_out)
pickle_out.close()
