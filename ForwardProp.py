import numpy as np
import skimage.io
import sys
import math
from PIL import Image

# def load_and_pad_input(image_path, padding=2, padded_number=0):
#     raw_img = Image.open(image_path, 'r')
#     mat_img = np.transpose(list(raw_img.getdata())) \
#                 .reshape(3, raw_img.size[0], raw_img.size[1])

#     return pad_matrix(mat_img, padding, padded_number)

# def pad_matrix(mat_img, padding, padded_number):
#     def pad_with(vector, pad_width, iaxis, kwargs):
#         pad_value = kwargs.get('padder', padded_number)
#         vector[:pad_width[0]] = pad_value
#         vector[-pad_width[1]:] = pad_value

#     return np.pad(
#         mat_img, 
#         ((0, 0), (padding, padding), (padding, padding)), 
#         pad_with, 
#         padder=padded_number
#     )

def load_image(image_path):
    raw_img = skimage.io.imread(image_path)
    return raw_img

"""
padding()
Params:
    - image_path: string; path to image relative to current directory
    - pad_layer: int; number of padding layers
    - padded_number: numeric; the number to be padded
"""
def padding(input_matrix, pad_layer=2, padded_number=0):
    if (len(input_matrix.shape) > 2):
        padded_result = np.zeros((input_matrix.shape[0]+pad_layer*2, input_matrix.shape[1]+pad_layer*2, input_matrix.shape[2]))
        for i in range(input_matrix.shape[-1]):
            current_channel = input_matrix[:, :, i]
            padded_current_channel = np.pad(current_channel, pad_layer, mode='constant', constant_values=padded_number)
            padded_result[:, :, i] = padded_current_channel
    else:
        padded_result = np.pad(input_matrix, pad_layer, mode='constant', constant_values=padded_number)

    return padded_result

def conv_(input_matrix, conv_filter, stride):
    filter_size = conv_filter.shape[1]
    result = np.zeros((input_matrix.shape))
    #Looping through the image to apply the convolution operation.
    # iterate row
    for r in np.uint16(np.arange(
        filter_size/2.0, 
        input_matrix.shape[0]-filter_size/2.0+1,
        stride
        )):
        # iterate column
        for c in np.uint16(np.arange(
            filter_size/2.0, 
            input_matrix.shape[1]-filter_size/2.0+1,
            stride
            )):
            curr_region = input_matrix[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                            c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]

            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)
            result[r, c] = conv_sum
            
    #Clipping the outliers of the result matrix.
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), 
                        np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    
    return final_result

def init_filter(filter_number, filter_size, filter_channel, input_dim, input_channel):
    #Initialize Convulation Filter
    if (input_dim > 2):
        filter_channel = input_channel
        conv_filter = np.zeros((filter_number, filter_size, filter_size, filter_channel))
        for i in range(filter_number):
            conv_filter[i, :, :, :] = np.random.uniform(low=-0.1, high=0.1, size=(filter_size, filter_size, filter_channel))
    else:
        conv_filter = np.zeros((filter_number, filter_size, filter_size))
        for i in range(filter_number):
            conv_filter[i, :, :] = np.random.uniform(low=-0.1, high=0.1, size=(filter_size, filter_size))
    
    return conv_filter
    
def conv(input_matrix, filter_number, filter_size, pad_layer, padded_number, stride):

    #Apply padding
    input_matrix = padding(input_matrix, pad_layer, padded_number)

    #Initialize Convulation Filter
    if (len(input_matrix.shape) > 2):
        filter_channel = input_matrix.shape[-1]
        conv_filter = np.zeros((filter_number, filter_size, filter_size, filter_channel))
        for i in range(filter_number):
            conv_filter[i, :, :, :] = np.random.uniform(low=-0.1, high=0.1, size=(filter_size, filter_size, filter_channel))
    else:
        conv_filter = np.zeros((filter_number, filter_size, filter_size))
        for i in range(filter_number):
            conv_filter[i, :, :] = np.random.uniform(low=-0.1, high=0.1, size=(filter_size, filter_size))

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((input_matrix.shape[0]-conv_filter.shape[1]+1, 
                                input_matrix.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))

    if (len(input_matrix.shape) > 2):
        filter_channel = input_matrix.shape[-1]
        conv_filter = np.zeros((filter_number,filter_size,filter_size,filter_channel))
        for i in range(filter_number):
            conv_filter[i, :, :, :] = np.random.uniform(low=-0.1, high=0.1,size=(filter_size,filter_size,filter_channel))
    else:
        conv_filter = np.zeros((filter_number,filter_size,filter_size))
        for i in range(filter_number):
            conv_filter[i, :, :] = np.random.uniform(low=-0.1,high=0.1,size=(filter_size,filter_size))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.

        if len(curr_filter.shape) > 2:
            conv_map = conv_(input_matrix[:, :, 0], curr_filter[:, :, 0], stride) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(input_matrix[:, :, ch_num], 
                                  curr_filter[:, :, ch_num], stride)
        else: # There is just a single channel in the filter.
            conv_map = conv_(input_matrix, curr_filter)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.
    

def pooling(feature_map, mode='max', size=2, stride=2):
    #Preparing the output of the pooling operation.
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride+1),
                            np.uint16((feature_map.shape[1]-size+1)/stride+1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size+1, stride):
                if(mode=='max'):
                    pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size,  c:c+size, map_num]])
                else:
                    pool_out[r2, c2, map_num] = np.mean([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out

def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0,feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out

def flatten(output_layer):
    flatten_result = np.ravel(output_layer)
    return flatten_result

def dense(dense_input, class_num, activation_func="sigmoid"):
    def sigmoid_dense(net):
        return np.float64(1 / (1 + np.exp(-net)))

    def relu_dense(x):
        if(x < 0):
            return 0
        return x

    # create matrix of size (dense_input.size, class_num)
    input_size = dense_input.size
    flattened_input = dense_input.reshape(input_size)
    weights = np.random.uniform(-1, 1, (input_size, class_num))
    output = np.zeros(class_num)

    for w in range(input_size):
        for c in range(class_num):
            output[c] += flattened_input[w] * weights[w][c] 

    # activation function defaults to sigmoid
    if(activation_func=="relu"):
        vectorized_activation = np.vectorize(relu_dense)
    else:
        vectorized_activation = np.vectorize(sigmoid_dense)

    return vectorized_activation(output)