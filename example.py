import skimage.data
import numpy
import matplotlib
import matplotlib.pyplot
import ForwardProp as numpycnn

"""
Convolutional neural network implementation using NumPy
An article describing this project is titled "Building Convolutional Neural Network using NumPy from Scratch". It is available in these links: https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad/
https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741

The project is tested using Python 3.5.2 installed inside Anaconda 4.2.0 (64-bit)
NumPy version used is 1.14.0

For more info., contact me:
    Ahmed Fawzy Gad
    KDnuggets: https://www.kdnuggets.com/author/ahmed-gad
    LinkedIn: https://www.linkedin.com/in/ahmedfgad
    Facebook: https://www.facebook.com/ahmed.f.gadd
    ahmed.f.gad@gmail.com
    ahmed.fawzy@ci.menofia.edu.eg
"""

# Reading the image
#img = skimage.io.imread("test.jpg")
#img = skimage.data.checkerboard()
img = skimage.data.chelsea()
#img = skimage.data.camera()

# Converting the image into gray.

# First conv layer
#l1_filter = numpy.random.rand(2,7,7)*20 # Preparing the filters randomly.

l1_filter_number = int(input("Number of filter = "))
l1_filter_size = int(input("Filter size = "))
l1_filter_channel = int(input("Filter channel = "))
l1_pool_size = int(input("Pool size = "))
l1_pool_stride = int(input("Pooling stride = "))

l1_filter = numpy.zeros((l1_filter_number,l1_filter_size,l1_filter_size,l1_filter_channel))

for i in range(l1_filter_number):
    l1_filter[i, :, :, :] = numpy.random.rand(1,l1_filter_size,l1_filter_size,l1_filter_channel)


#Convoluting the image

print("\n**Working with conv layer 1**")
l1_feature_map = numpycnn.conv(img, l1_filter)
print("\n**ReLU**")
l1_feature_map_relu = numpycnn.relu(l1_feature_map)
print("\n**Pooling**")
l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, l1_pool_size, l1_pool_size)
print("**End of conv layer 1**\n")

# Graphing results
fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
ax0.imshow(img)
ax0.set_title("Input Image")
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
matplotlib.pyplot.close(fig0)

# Layer 1


fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols = l1_feature_map.shape[2])
print("nocols:" + str(l1_feature_map.shape[2]))

for i in range(l1_feature_map.shape[2]):
    ax1[0, i].imshow(l1_feature_map[:, :, i])
    ax1[0, i].get_xaxis().set_ticks([])
    ax1[0, i].get_yaxis().set_ticks([])
    ax1[0, i].set_title("L1-Convolution " + str(i + 1))

for i in range(l1_feature_map_relu.shape[2]):
    ax1[1, i].imshow(l1_feature_map_relu[:, :, i])
    ax1[1, i].get_xaxis().set_ticks([])
    ax1[1, i].get_yaxis().set_ticks([])
    ax1[1, i].set_title("L1-Detector " + str(i + 1))

for i in range(l1_feature_map_relu_pool.shape[2]):
    ax1[2, i].imshow(l1_feature_map_relu_pool[:, :, i])
    ax1[2, i].get_xaxis().set_ticks([])
    ax1[2, i].get_yaxis().set_ticks([])
    ax1[2, i].set_title("L1-Pooling" + str(i + 1))


matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
matplotlib.pyplot.close(fig1)
