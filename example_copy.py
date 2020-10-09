import numpy
import matplotlib
import matplotlib.pyplot
import CNNClass
from utils import load_and_resize_image, get_data
from sklearn.metrics import accuracy_score

TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
INPUT_SHAPE = (100, 100, 3)

# Reading the image
train_names, train_val, train_paths = get_data(TRAIN_DIR)
test_names, test_val, test_paths = get_data(TEST_DIR)

train_images = []
test_images = []

for img in train_paths:
    train_images.append(load_and_resize_image(img, INPUT_SHAPE))

for img in test_paths:
    test_images.append(load_and_resize_image(img, INPUT_SHAPE))

# Define model
model = CNNClass.Sequential(input_shape=INPUT_SHAPE)

# Convoluting the image
print("**Convolution Layer Start**")
print("**Convolution Stage**")

model.add(CNNClass.Conv2D(2, 3, 2, 2, 0, 2))

print("**Detector Stage**")
model.add(CNNClass.Activation("relu"))

print("**Pooling Stage**")
model.add(CNNClass.Pooling(2, 2, 2))

print("**End of Convolution Layer**\n")

print("**Dense layer**")
model.add(CNNClass.Flatten())

model.add(CNNClass.Dense(2))
model.add(CNNClass.Activation("sigmoid", class_num=2))
print("**Dense output**")

model.train(train_images, train_val, epochs=10)
pred = model.predict(test_images)
print("Test Predictions\n", numpy.array(pred))
print("Accuracy\n", accuracy_score(test_val, pred))