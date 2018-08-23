from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.utils import np_utils

from my_models import MyModels

import random
import numpy as np


random.seed(42)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0],28, 28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],28, 28,1).astype('float32')


# build the model
model = MyModels.baseline_model(width=28,height=28,depth=1,classes=10)

#compile the model
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)

print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save("mnist.h5")