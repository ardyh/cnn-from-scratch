import numpy as np

class Input:
    def __init__(self, input_shape)
        self.input_size = input_shape
        self.layer_output_size = input_shape

class Convolutional_Layer:
    def __init__(self, previous_layer, n_filter, size_filter, padding, stride):
        self.n_filter = n_filter
        self.size_filter = size_filter
        self.padding = padding
        self.stride = stride
        self.previous_layer = previous_layer

        self.layer_input_size = (self.previous_layer.layer_output_size[0] + 2 * self.padding, self.previous_layer.layer_output_size[1] + 2 * self.padding, self.previous_layer.layer_output_size[2])
        self.layer_output_size = (((self.previous_layer.layer_output_size[0] - self.size + 2 * self.padding) / self.stride + 1),
                                ((self.previous_layer.layer_output_size[1] - self.size + 2 * self.padding) / self.stride + 1),
                                self.n_filter)

    def convolution_process(self, layer_input):
        #Add padding to layer_input
        layer_input = layer_input.pad(layer_input, ((self.padding,self.padding),(self.padding,self.padding)), 'constant')


class Detector_Stage:
    def __init__(self, previous_layer):
        self.previous_layer = previous_layer
        self.layer_input_size = self.previous_layer.layer_output_size
        self.layer_output_size = self.previous_layer.layer_output_size

    def relu_process(self, layer_input)
        self.layer_output = np.zeros(self.layer_output_size)
        for layer_idx in range(self.layer_input.shape[-1]):
            for r in np.arange(0,self.layer_input.shape[0]):
                for c in np.arange(0, self.layer_input.shape[1]):
                    self.layer_output[r, c, layer_idx] = np.max([self.feature_map[r, c, layer_idx], 0])

class Pooling_Stage:
    def __init__(self, previous_layer, size, stride, mode):
        self.size = size
        self.stride = stride
        self.mode = mode
        self.previous_layer = previous_layer
        self.layer_input_size = self.previous_layer.layer_output_size
        self.layer_output_size = (((self.previous_layer.layer_output_size[0] - self.size) / self.stride + 1),
                                ((self.previous_layer.layer_output_size[1] - self.size) / self.stride + 1), 1)

    def pool_process(self, layer_input):
        result = np.zeros(self.layer_output_size)
        for map_num in range(layer_input.shape[-1]):
            r2 = 0
            for r in numpy.arange(0,layer_input.shape[0]-self.size+1, self.stride):
                c2 = 0
                for c in numpy.arange(0, layer_input.shape[1]-self.size+1, self.stride):
                    if self.mode == 'max':
                        result[r2, c2, map_num] = numpy.max([layer_input[r:r+self.size,  c:c+self.pool_size, map_num]])
                    elif self.mode == 'average':
                        result[r2, c2, map_num] = numpy.mean([layer_input[r:r+self.size,  c:c+self.pool_size, map_num]])
                    c2 = c2 + 1
                r2 = r2 +1

        self.layer_output = result

class Flatten:
    def __init__(self, previous_layer):
        self.previous_layer = previous_layer
        self.layer_input_size = self.previous_layer.layer_output_size

    def flatten_process(self, layer_input):
        self.layer_output_size = (layer_input.size, 1, 1)
        self.layer_output = np.ravel(layer_input)

class Dense:
    def __init__(self, previous_layer):
        self.previous_layer = previous_layer
        self.layer_input_size = self.previous_layer.layer_output_size