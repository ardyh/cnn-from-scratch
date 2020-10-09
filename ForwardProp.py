import numpy as np
import skimage.io
from utils import load_image


def add_padding(input_matrix, padding_layer, padding_number):
    if (len(input_matrix.shape) > 2):
        #Initialize matrix result
        padded_result = np.zeros((input_matrix.shape[0] + padding_layer * 2, 
                                  input_matrix.shape[1] + padding_layer * 2,
                                  input_matrix.shape[-1]))

        for i in range(input_matrix.shape[-1]):
            current_channel = input_matrix[:, :, i]
            padded_current_channel = np.pad(current_channel, padding_layer, mode='constant', constant_values=padding_number)
            padded_result[:, :, i] = padded_current_channel
    else:
        padded_result = np.pad(input_matrix, padding_layer, mode='constant', constant_values=padding_number)

    return padded_result

def convolution_calculation(input_matrix, conv_filter, padding_layer, stride):
    #input_matrix shape = (mxn)

    #Initialize matrix result
    output_shape = (((input_matrix.shape[0] - conv_filter.shape[0] + 2 * padding_layer) // stride + 1),
                    ((input_matrix.shape[1] - conv_filter.shape[1] + 2 * padding_layer) // stride + 1))
    output_matrix = np.zeros(output_shape)

    input_length_idx = 0; output_length_idx = 0
    while(input_length_idx < input_matrix.shape[0] - conv_filter.shape[0] + 1):
        
        input_width_idx = 0; output_width_idx = 0
        while(input_width_idx < input_matrix.shape[1] - conv_filter.shape[1] + 1):
            # Get receptive field
            current_region = input_matrix[input_length_idx : (input_length_idx + conv_filter.shape[0]),
                                          input_width_idx : (input_width_idx + conv_filter.shape[1])]
            
            # Get sum of (dot product of receptive field and filter)
            product_result = current_region * conv_filter
            sum_result = np.sum(product_result)

            output_matrix[output_length_idx, output_width_idx] = sum_result

            input_width_idx += stride
            output_width_idx +=  1
        #End while
        input_length_idx += stride
        output_length_idx += 1
    #End while

    return output_matrix

def init_filter(filter_number, filter_size_length, filter_size_width, input_shape):
    # set the lower and upper bound of randomized parameters to be intialized in filter
    LOWER_BOUND = -10
    UPPER_BOUND = 10

    #Initialize Convulation Filter / Kernel
    if (len(input_shape) > 2):
        filter_channel = input_shape[-1]

        convolution_filter = np.zeros((filter_number, filter_size_length, filter_size_width, filter_channel))
        for i in range(filter_number):
            convolution_filter[i, :, :, :] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(filter_size_length, filter_size_width, filter_channel))
    else:
        convolution_filter = np.zeros((filter_number, filter_size_length, filter_size_width))
        for i in range(filter_number):
            convolution_filter[i, :, :] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(filter_size_length, filter_size_width))
    
    return convolution_filter
    
def convolution_stage(input_matrix, filter_number, filter_size_length, filter_size_width, padding_layer=0, padded_number=0, stride=1):
    #Stage Validation
    if (len(input_matrix.shape) < 2):
        raise Exception("Invalid input matrix")
    
    if (filter_number <= 0 or filter_size_length <= 0 or filter_size_width <= 0 or stride <= 0 or padding_layer < 0) :
        raise Exception("Error input parameter")
    
    #Apply padding
    input_matrix = add_padding(input_matrix, padding_layer, padded_number)

    #Initialize Convulation Filter
    conv_filter = init_filter(filter_number, filter_size_length, filter_size_width, input_matrix.shape)

    #Initialize bias with 0
    list_bias = np.random.uniform(-1, 1, conv_filter.shape[0])

    #Feature map formula : (W - F + 2P) / S + 1
    feature_map_shape = (((input_matrix.shape[0] - conv_filter.shape[1] + 2 * padding_layer) // stride + 1),
                          ((input_matrix.shape[1] - conv_filter.shape[2] + 2 * padding_layer) // stride + 1),
                          filter_number)
    
    # Initialize feature map output
    feature_map = np.zeros(feature_map_shape)

    for filter_num in range(conv_filter.shape[0]):
        current_filter = conv_filter[filter_num, :]

        if len(current_filter.shape) > 2:
            convolution_result = convolution_calculation(input_matrix[:, :, 0], current_filter[:, :, 0], padding_layer, stride)
            for channel in range(1, current_filter.shape[-1]):
                convolution_result = convolution_result + convolution_calculation(input_matrix[:, :, channel], current_filter[:, :, channel], padding_layer, stride)
        else:
            convolution_result = convolution_calculation(input_matrix, current_filter, padding_layer, stride)
        
        feature_map[:, :, filter_num] = convolution_result + list_bias[filter_num]
    return feature_map

def pool_stage(input_matrix, pool_length, pool_width, stride=2, mode='max'):
    # Stage Validation
    if (len(input_matrix.shape) < 2):
        raise Exception("Invalid input matrix")
    
    if (pool_length <= 0 or pool_width <= 0 or stride <= 0):
        raise Exception ("Error input parameter")
    
    #Initialize matrix result
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
                current_region = region[input_length_idx : (input_length_idx + pool_length),
                                        input_width_idx : (input_width_idx + pool_width)]

                if mode == 'max':
                    curr_result = np.max(current_region)
                elif mode == 'average':
                    curr_result = np.mean(current_region)
                else:
                    raise Exception("Invalid mode")

                output_matrix[output_length_idx, output_width_idx] = curr_result

                input_width_idx += stride
                output_width_idx +=  1
            #End while
            input_length_idx += stride
            output_length_idx += 1
        #End while
        
        result[:,:,channel] = output_matrix

    return result

def sigmoid(net):
    return 1. / (1. + np.exp(-net))

def relu(net):
    return 0 if (net < 0) else net

def activation(input_matrix, function_name="relu"):
    # activation function defaults to relu
    if (function_name == "sigmoid"):
        v_activation = np.vectorize(sigmoid)
    elif (function_name == "relu"):
        v_activation = np.vectorize(relu)
    else:
        raise Exception("Invalid activation function")

    return v_activation(input_matrix)

def flatten(output_layer):
    return np.ravel(output_layer)

def dense(dense_input, class_num, func_name="relu"):
    # create matrix of size (dense_input.size, class_num)

    #validation
    if class_num <= 0 :
        raise Exception("Invalid class number")
    input_size = dense_input.size
    flattened_input = dense_input.reshape(input_size)
    weights = np.random.uniform(-1, 1, (input_size, class_num))
    output = np.zeros(class_num)

    for w in range(input_size):
        for c in range(class_num):
            output[c] += flattened_input[w] * weights[w][c] 

    # activation function defaults to relu
    return activation(output, func_name)