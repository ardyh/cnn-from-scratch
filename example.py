import skimage.data
import numpy
import matplotlib
import matplotlib.pyplot
import ForwardProp as numpycnn
import CNNClass

BASE_IMG_DIR = "images/sandbox/"

# Reading the image
img = numpycnn.load_image(f"{BASE_IMG_DIR}cat.38.jpg")

# Define model
model = CNNClass.Sequential()

# Convoluting the image
print("**Convolution Layer Start**")
print("**Convolution Stage**")
# Prev
# l1_feature_map = numpycnn.convolution_stage(img, 2, 3, 2, 2, 0, 2) # (input_matrix, filter_number, filter_size_length, filter_size_width, pad_layer, padded_number, stride=1):
# After
model.add(CNNClass.Conv2D(2, 3, 2, 2, 0, 2))

print("**Detector Stage**")
# Prev
# l1_feature_map_relu = numpycnn.activation(l1_feature_map)
# After
model.add(CNNClass.Activation("relu"))

print("**Pooling Stage**")
# Prev
# l1_feature_map_relu_pool = numpycnn.pool_stage(l1_feature_map_relu, 2, 2, 2)
# After
model.add(CNNClass.Pooling(2, 2, 2))

print("**End of Convolution Layer**\n")

print("**Dense layer**")
# Prev
# l1_flat = numpycnn.flatten(l1_feature_map_relu_pool)
# After
model.add(CNNClass.Flatten())

# Prev
# l1_dense_output = numpycnn.dense(l1_flat, 10, "relu")
# After
model.add(CNNClass.Dense(2))
model.add(CNNClass.Activation("sigmoid"))
print("**Dense output**")
print("weights before update")
print(model.layers[-2].weights)

# Prev
# print(l1_dense_output)

# After
# FORWARD PROP
model.forwardprop(img)
# COBA COBA
model.layers[-1].calculate_delta_output(numpy.array([0, 1])) 
model.layers[-2].prev_error = model.layers[-1].passed_error
model.layers[-2].calculate_error()
model.layers[-2].update_weight()
print("weights after update")
print(model.layers[-2].weights)
model_output = model.final_output
# SIMPEN HASIL
model_feature_layer_output = model.feature_layer_output
print("CNN CLASS MODEL OUTPUT")
print(model_output)

# Showing the image output of each layer
fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols = model_feature_layer_output[0].shape[2])

for i in range(model_feature_layer_output[0].shape[2]):
    ax1[0, i].imshow(model_feature_layer_output[0][:, :, i])
    ax1[0, i].get_xaxis().set_ticks([])
    ax1[0, i].get_yaxis().set_ticks([])
    ax1[0, i].set_title("L1-Convolution " + str(i + 1))

for i in range(model_feature_layer_output[1].shape[2]):
    ax1[1, i].imshow(model_feature_layer_output[1][:, :, i])
    ax1[1, i].get_xaxis().set_ticks([])
    ax1[1, i].get_yaxis().set_ticks([])
    ax1[1, i].set_title("L1-Detector " + str(i + 1))

for i in range(model_feature_layer_output[2].shape[2]):
    ax1[2, i].imshow(model_feature_layer_output[2][:, :, i])
    ax1[2, i].get_xaxis().set_ticks([])
    ax1[2, i].get_yaxis().set_ticks([])
    ax1[2, i].set_title("L1-Pooling" + str(i + 1))


matplotlib.pyplot.savefig(f"{BASE_IMG_DIR}sample_layer_output.png", bbox_inches="tight")
matplotlib.pyplot.close(fig1)
