import skimage.data
import numpy
import matplotlib
import matplotlib.pyplot
import ForwardProp as numpycnn

BASE_IMG_DIR = "images/sandbox/"

# Reading the image
img = numpycnn.load_image(f"{BASE_IMG_DIR}cat.38.jpg")

#Convoluting the image
print("\n**Working with conv layer**")
l1_feature_map = numpycnn.convolution_stage(img, 2, 3, 2, 2, 0, 2) # (input_matrix, filter_number, filter_size_length, filter_size_width, pad_layer, padded_number, stride=1):
print("\n**ReLU**")
l1_feature_map_relu = numpycnn.activation(l1_feature_map)
print("\n**Pooling**")
l1_feature_map_relu_pool = numpycnn.pool_stage(l1_feature_map_relu, 2, 2, 2)
print("**End of conv layer**\n")

print("**Dense layer**")
l1_flat = numpycnn.flatten(l1_feature_map_relu_pool)
l1_dense_output = numpycnn.dense(l1_flat, 10, "sigmoid")

print("\n**Dense output**")
print(l1_dense_output)

# Showing the image output of each layer
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


matplotlib.pyplot.savefig(f"{BASE_IMG_DIR}sample_layer_output.png", bbox_inches="tight")
matplotlib.pyplot.close(fig1)
