# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# load the image
image = cv2.imread("six.png")

 
# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = imutils.rotate(image, 60)
orig = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model("mnist.h5")

# classify the input image
prediction = model.predict(image)

y_pred = prediction.tolist()

print(y_pred[0].index(max(y_pred[0])))


cv2.imshow("Orginal",orig)
cv2.waitKey(0)