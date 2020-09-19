import numpy as np
import skimage.io
import sys
import math
from PIL import Image

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

def conv_calculation(input_matrix, conv_filter, pad_layer, padded_number, stride=1):
    #input_matrix shape = (mxn)
    output_shape = (((input_matrix.shape[0] - conv_filter.shape[0] + 2 * pad_layer) // stride + 1),
                    ((input_matrix.shape[1] - conv_filter.shape[1] + 2 * pad_layer) // stride + 1))

    output_matrix = np.zeros(output_shape)

    input_length_idx = 0; output_length_idx = 0
    while(input_length_idx < input_matrix.shape[0] - conv_filter.shape[0] + 1):
        
        input_width_idx = 0; output_width_idx = 0
        while(input_width_idx < input_matrix.shape[1] - conv_filter.shape[1] + 1):
            # Get receptive field
            curr_region = input_matrix[input_length_idx : (input_length_idx + conv_filter.shape[0]),
                                       input_width_idx : (input_width_idx + conv_filter.shape[1])]
            
            # Get sum of (dot product of receptive field and filter)
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)

            output_matrix[output_length_idx, output_width_idx] = conv_sum

            input_width_idx += stride
            output_width_idx +=  1
        #End while
        input_length_idx += stride
        output_length_idx += 1
    #End while

    return output_matrix

def init_filter(filter_number, filter_size_length, filter_size_width, input_dim, input_channel):
    #Initialize Convulation Filter
    if (input_dim > 2):
        filter_channel = input_channel
        conv_filter = np.zeros((filter_number, filter_size_length, filter_size_width, filter_channel))
        for i in range(filter_number):
            conv_filter[i, :, :, :] = np.random.uniform(low=-0.1, high=0.1, size=(filter_size_length, filter_size_width, filter_channel))
    else:
        conv_filter = np.zeros((filter_number, filter_size_length, filter_size_width))
        for i in range(filter_number):
            conv_filter[i, :, :] = np.random.uniform(low=-0.1, high=0.1, size=(filter_size_length, filter_size_width))
    
    return conv_filter
    
def conv(input_matrix, filter_number, filter_size_length, filter_size_width, pad_layer, padded_number, stride=1):
    #Apply padding
    input_matrix = padding(input_matrix, pad_layer, padded_number)

    #Initialize Convulation Filter
    conv_filter = init_filter(filter_number, filter_size_length, filter_size_width, len(input_matrix.shape), input_matrix.shape[-1])

    #Initialize bias
    list_bias = np.zeros(conv_filter.shape[0])

    #Feature map formula : (W - F + 2P) / S + 1
    feature_maps_shape = (((input_matrix.shape[0] - conv_filter.shape[1] + 2 * pad_layer) // stride + 1),
                          ((input_matrix.shape[1] - conv_filter.shape[2] + 2 * pad_layer) // stride + 1),
                          filter_number)
    
    # Initialize feature map output
    feature_maps = np.zeros(feature_maps_shape)

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.

        if len(curr_filter.shape) > 2:
            conv_map = conv_calculation(input_matrix[:, :, 0], curr_filter[:, :, 0], pad_layer, padded_number, stride) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_calculation(input_matrix[:, :, ch_num], curr_filter[:, :, ch_num], pad_layer, padded_number, stride)
        else: # There is just a single channel in the filter.
            conv_map = conv_calculation(input_matrix, curr_filter, pad_layer, padded_number, stride)
        
        feature_maps[:, :, filter_num] = conv_map + list_bias[filter_num] # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.

def pool(input_matrix, pool_length, pool_width, stride, mode='max'):
    result_shape = (((input_matrix.shape[0] - pool_length) // stride + 1),
                    ((input_matrix.shape[1] - pool_width) // stride + 1),
                    input_matrix.shape[-1])
    
    result = np.zeros(result_shape)
    
    for channel in range(input_matrix.shape[-1]):
        #input_matrix shape = (mxn)
        region = input_matrix[:,:,channel]
        output_shape = (((region.shape[0] - pool_length) // stride + 1),
                        ((region.shape[1] - pool_width) // stride + 1))
        
        output_matrix = np.zeros(output_shape)

        input_length_idx = 0; output_length_idx = 0
        while(input_length_idx < region.shape[0] - pool_length + 1):

            input_width_idx = 0; output_width_idx = 0
            while(input_width_idx < region.shape[1] - pool_width + 1):
                curr_region = region[input_length_idx : (input_length_idx + pool_length),
                                           input_width_idx : (input_width_idx + pool_width)]

                if mode == 'max':
                    curr_result = np.max(curr_region)
                else:
                    curr_result = np.mean(curr_region)

                output_matrix[output_length_idx, output_width_idx] = curr_result

                input_width_idx += stride
                output_width_idx +=  1
            #End while
            input_length_idx += stride
            output_length_idx += 1
        #End while
        
        result[:,:,channel] = output_matrix

    return result

def relu(feature_map):
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
        return 1. / (1. + np.exp(-net))

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