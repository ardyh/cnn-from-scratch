import skimage.data
import numpy
import matplotlib
import matplotlib.pyplot
import ForwardProp as numpycnn

# Reading the image
#img = skimage.io.imread("test.jpg")
# img = skimage.data.checkerboard()\
# img = skimage.data.chelsea()
#img = skimage.data.camera()

# Converting the image into gray.

# First conv layer
#l1_filter = numpy.random.rand(2,7,7)*20 # Preparing the filters randomly.

l1_filter_number = int(input("Number of filter = "))
l1_filter_size = int(input("Filter size = "))
l1_pool_size = int(input("Pool size = "))
l1_pool_stride = int(input("Pooling stride = "))
l1_padding = int(input("Padding = "))

img = numpycnn.load_and_pad_input2("images/sandbox/cat.38.jpg", l1_padding)

if (len(img.shape) > 2):
    l1_filter_channel = img.shape[-1]
    l1_filter = numpy.zeros((l1_filter_number,l1_filter_size,l1_filter_size,l1_filter_channel))
    for i in range(l1_filter_number):
        l1_filter[i, :, :, :] = numpy.random.rand(1,l1_filter_size,l1_filter_size,l1_filter_channel)
else:
    l1_filter = numpy.zeros((l1_filter_number,l1_filter_size,l1_filter_size))
    for i in range(l1_filter_number):
        l1_filter[i, :, :] = numpy.random.rand(1,l1_filter_size,l1_filter_size)


#Convoluting the image

print("\n**Working with conv layer 1**")
l1_feature_map = numpycnn.conv(img, l1_filter)
print("\n**ReLU**")
l1_feature_map_relu = numpycnn.relu(l1_feature_map)
print("\n**Pooling**")
l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, l1_pool_size, l1_pool_size)
print("**End of conv layer 1**\n")


# Layer 1


fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols = l1_feature_map.shape[2])

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
