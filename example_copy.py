import numpy
import matplotlib
import matplotlib.pyplot
import ForwardProp as numpycnn
import CNNClass
from utils import load_image, get_data

TRAIN_DIR = "train"
TEST_DIR = "test"

print(numpy.argmax(numpy.array([0,5])))

# Reading the image
train_names, train_val, train_paths = get_data(TRAIN_DIR)
test_names, test_val, test_paths = get_data(TEST_DIR)

train_images = []
test_images = []

for img in train_paths:
    train_images.append(load_image(img))

for img in test_paths:
    test_images.append(load_image(img))

print(train_images[0].shape)
# Define model
model = CNNClass.Sequential()

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
model.add(CNNClass.Activation("sigmoid"))
print("**Dense output**")

model.train(train_images, train_val, batch_size =1)
model.predict(test_images)
