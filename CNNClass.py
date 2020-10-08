import numpy as np
import skimage.io

class Sequential:
    def __init__(self, n_epoch=50, learning_rate=0.5, momentum=0.1):
        self.input = []
        self.layers = [] 
        # DEBUG
        self.conv_layer_final_idx = 3
        self.feature_layer_output = []
        # DEBUG END
        self.final_output = []
        
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.momentum = momentum

    def add(self, layer):
        self.layers.append(layer)
        return None

    def forwardprop(self, X_instance):
        prev_output = [] 
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.input = X_instance
            else:
                layer.input = prev_output
                
            layer.init_params(layer.input.shape)
            layer.run()
            prev_output = layer.output

            # DEBUG
            if idx < self.conv_layer_final_idx:
                self.feature_layer_output.append(layer.output)
            # DEBUG END
        
        self.final_output = prev_output

    def backprop(self, y_instance, is_update):
        prev_error = []; 
        error_calc_output = []
        for idx, layer in enumerate(reversed(self.layers)):
            if idx == 0: # If last layer
                layer.error_calc_input = y_instance
            else:
                layer.error_calc_input = error_calc_output
                layer.prev_error = prev_error
            
            prev_error = layer.passed_error
            error_calc_output = layer.error_calc_output

    def train(self, X, y, batch_size):
        instance_size = len(X)
        
        # Initialize parameters for every layer
        sample_instance = X[0]
        for layer in self.layers:
            layer.init_params(X[0].shape)

        # iterate every epoch
        for epoch in list(range(self.n_epoch)):
            # iterate every instance
            for instance_idx, instance in enumerate(zip(X, y)):
                X_instance = instance[0]; y_instance = instance[1]
                
                # If last instance or multiple of batch, update params
                is_update = False
                if ((instance_idx == instance_size) or (instance_idx % batch_size == 0)):
                    is_update = True
                
                self.forwardprop(X_instance)
                self.backprop(y_instance, is_update)
        return None

    def predict(self, X):
        return None
    
# Conv Layer
class Conv2D:
    def __init__(self, filter_number, filter_size_length, filter_size_width, n_channels=3, padding_layer=0, padded_number=0, stride=1):
        self.input = []
        self.output = []
        
        self.n_channels = n_channels
        self.filter = []
        self.filter_number = filter_number
        self.filter_size_length = filter_size_length
        self.filter_size_width = filter_size_width
        self.padding_layer = padding_layer
        self.padded_number = padded_number
        self.stride = stride

    def init_params(self, input_shape):
        self.filter = self.init_filter(
            self.filter_number, 
            self.filter_size_length, 
            self.filter_size_width, 
            input_shape
            )

    def add_padding(self, input_matrix, padding_layer, padding_number):
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


    def convolution_calculation(self, input_matrix, conv_filter, padding_layer, stride):
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

    # def init_filter(self, filter_number, filter_size_length, filter_size_width, input_shape):
    # Asumsi selalu 2D, maka input_shape ga butuh
    def init_filter(self, filter_number, filter_size_length, filter_size_width, input_shape):
        # set the lower and upper bound of randomized parameters to be intialized in filter
        LOWER_BOUND = -10
        UPPER_BOUND = 10

        convolution_filter = []
        filter_channel = input_shape[-1]

        convolution_filter = np.zeros((filter_number, filter_size_length, filter_size_width, filter_channel))
        for i in range(filter_number):
            convolution_filter[i, :, :, :] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(filter_size_length, filter_size_width, filter_channel))
        
        #Initialize Convulation Filter / Kernel
        # if (len(input_shape) > 2):
        #     filter_channel = input_shape[-1]

        #     convolution_filter = np.zeros((filter_number, filter_size_length, filter_size_width, filter_channel))
        #     for i in range(filter_number):
        #         convolution_filter[i, :, :, :] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(filter_size_length, filter_size_width, filter_channel))
        # else:
        #     convolution_filter = np.zeros((filter_number, filter_size_length, filter_size_width))
        #     for i in range(filter_number):
        #         convolution_filter[i, :, :] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(filter_size_length, filter_size_width))
        
        return convolution_filter
    

    def run(self):
        #Stage Validation
        if (len(self.input.shape) < 2):
            raise Exception("Invalid input matrix")
        
        if (self.filter_number <= 0 or self.filter_size_length <= 0 or self.filter_size_width <= 0 or self.stride <= 0 or self.padding_layer < 0) :
            raise Exception("Error input parameter")
        
        #Apply padding
        self.input = self.add_padding(self.input, self.padding_layer, self.padded_number)

        #Initialize bias with 0
        list_bias = np.random.uniform(-1, 1, self.filter.shape[0])

        #Feature map formula : (W - F + 2P) / S + 1
        feature_map_shape = (((self.input.shape[0] - self.filter.shape[1] + 2 * self.padding_layer) // self.stride + 1),
                            ((self.input.shape[1] - self.filter.shape[2] + 2 * self.padding_layer) // self.stride + 1),
                            self.filter_number)
        
        # Initialize feature map output
        feature_map = np.zeros(feature_map_shape)

        for filter_num in range(self.filter.shape[0]):
            current_filter = self.filter[filter_num, :]

            if len(current_filter.shape) > 2:
                convolution_result = self.convolution_calculation(self.input[:, :, 0], current_filter[:, :, 0], self.padding_layer, self.stride)
                for channel in range(1, current_filter.shape[-1]):
                    convolution_result = convolution_result + self.convolution_calculation(self.input[:, :, channel], current_filter[:, :, channel], self.padding_layer, self.stride)
            else:
                convolution_result = self.convolution_calculation(self.input, current_filter, self.padding_layer, self.stride)
            
            feature_map[:, :, filter_num] = convolution_result + list_bias[filter_num]
        
        self.output = feature_map
        return None

    def backprop(self):
        pass

    def updateweight(self):
        pass

# Activation
class Activation:
    def __init__(self, function_name="relu"):
        self.input = []
        self.output = []
        self.function_name = function_name
        # Error calculation, backprop
        self.error_calc_input = [] # assume it's like y_true, but it can be propagated to other layers
        self.prev_error = []
        self.passed_error = []
        self.error = [] # harusnya gaperlu disini

    def init_params(self, input_shape):
        return None

    def calculate_delta_output(self, y_true):
        # Assuming untuk output layer
        for y_true_val, output_val in zip(y_true, self.output):
            class_error_term = 0
            if (self.function_name == "sigmoid"):
                # Buat hidden tinggal ambil term: output_val * (1 - output_val) 
                class_error_term = (y_true_val - output_val) * output_val * (1 - output_val) 
            elif (self.function_name == "relu"):
                # Buat hidden tinggal ambil term: relu_derivative_val
                relu_derivative_val = 1 if output_val > 0 else 0
                class_error_term = (y_true_val - output_val) * relu_derivative_val
            else:
                raise Exception("Invalid activation function name")
        
            self.passed_error.append(class_error_term) 
        
        self.passed_error = np.array(self.passed_error)

        # DEBUG
        print("self.output_act", self.output)
        print("self.passed_error_act", self.passed_error)
    
    # Assumption only works for layers where output dimension of next layer is equal to activation layer's input dimension
    def calculate_delta_hidden(self):
        # Mirip kaya calculate_delta_output, term yang digunakan sama, tapi perkalian matriks, bukan perkalian bilangan 
        # untuk setiap value di output, terapkan formula diatas

        # activation function defaults to relu
        if (function_name == "sigmoid"):
            v_activation = np.vectorize(self.d_sigmoid)
        elif (function_name == "relu"):
            v_activation = np.vectorize(self.d_relu)
        else:
            raise Exception("Invalid activation function name")

        curr_error = v_activation(self.output)
        
        # Asumsi: 
        #     - error dikali dari yang paling dekat output layer dulu
        #     - prev_error = error layer2 sebelumnya; jika di-iterasi mulai output layer 
        # Maka: delta = prev_error * curr_error; 
        
        # Cari dimensi yang sama dari prev_error
        if (self.prev_error.shape[1] == curr_error.shape[0]): #n_col(prev_error) == n_rows(curr_error), langsung kali
            self.error = np.matmul(self.prev_error, curr_error)
        elif (self.prev_error.shape[1] != curr_error.shape[0]): # Kalo n_col(prev_error) != n_rows(curr_error), transpose
            self.error = np.matmul(self.prev_error, curr_error.transpose())
        else: # if no intersection
            raise Exception("Cannot do matrix multiplication. No intersecting dimensions")

        return None

    def d_sigmoid(self, out):
        # Works for single value only
        return out * (1 - out)

    def d_relu(self, out):
        # Works for single value only
        return 1 if out > 0 else 0

    def sigmoid(self, net):
        return 1. / (1. + np.exp(-net))

    def relu(self, net):
        return 0 if (net < 0) else net

    def run(self):
        input_matrix = self.input
        function_name = self.function_name

        # activation function defaults to relu
        if (function_name == "sigmoid"):
            v_activation = np.vectorize(self.sigmoid)
        elif (function_name == "relu"):
            v_activation = np.vectorize(self.relu)
        else:
            raise Exception("Invalid activation function name")

        self.output = v_activation(input_matrix)
        return None

    def backward(self, error):
        result = np.zeros(self.output.shape)

        for channel in range(self.output.shape[-1]):
            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    if(self.output[i,j,channel] > 0):
                        result[i,j,channel] = error[i,j,channel]
                    else:
                        result[i,j,channel] = 0
        
        return result

# Pooling Layer
class Pooling:
    def __init__(self, pool_length, pool_width, stride=2, mode='max'):
        self.input = []
        self.output = []
        self.output_position_x = []
        self.output_position_y = []
        self.pool_length = pool_length 
        self.pool_width = pool_width 
        self.stride = stride 
        self.mode = mode 

    def init_params(self, input_shape):
        return None

    def run(self): 
        # Stage Validation
        if (len(self.input.shape) < 2):
            raise Exception("Invalid input matrix")
        
        if (self.pool_length <= 0 or self.pool_width <= 0 or self.stride <= 0):
            raise Exception ("Error input parameter")
        
        #Initialize matrix result
        result_shape = (((self.input.shape[0] - self.pool_length) // self.stride + 1),
                        ((self.input.shape[1] - self.pool_width) // self.stride + 1),
                        self.input.shape[-1])
        
        result = np.zeros(result_shape)
        result_position_x = np.zeros(result_shape)
        result_position_y = np.zeros(result_shape)
        
        for channel in range(self.input.shape[-1]):
            #self.input shape = (mxn)
            region = self.input[:,:,channel]
            output_shape = (((region.shape[0] - self.pool_length) // self.stride + 1),
                            ((region.shape[1] - self.pool_width) // self.stride + 1))
            
            output_matrix = np.zeros(output_shape)
            output_position_x_matrix = np.zeros(output_shape)
            output_position_y_matrix = np.zeros(output_shape)

            input_length_idx = 0; output_length_idx = 0
            while(input_length_idx < region.shape[0] - self.pool_length + 1):

                input_width_idx = 0; output_width_idx = 0
                while(input_width_idx < region.shape[1] - self.pool_width + 1):
                    current_region = region[input_length_idx : (input_length_idx + self.pool_length),
                                            input_width_idx : (input_width_idx + self.pool_width)]

                    if self.mode == 'max':
                        curr_result = np.max(current_region)
                        curr_position = np.argmax(current_region)
                    elif self.mode == 'average':
                        curr_result = np.mean(current_region)
                        curr_position = 0
                    else:
                        raise Exception("Invalid mode")

                    output_matrix[output_length_idx, output_width_idx] = curr_result
                    output_position_x_matrix[output_length_idx, output_width_idx] = input_length_idx + curr_position // current_region.shape[0]
                    output_position_y_matrix[output_length_idx, output_width_idx] = input_width_idx + curr_position % current_region.shape[0]

                    input_width_idx += self.stride
                    output_width_idx +=  1
                #End while
                input_length_idx += self.stride
                output_length_idx += 1
            #End while
            
            result[:,:,channel] = output_matrix
            result_position_x[:,:,channel] = output_position_x_matrix
            result_position_y[:,:,channel] = output_position_y_matrix

        self.output = result
        self.output_position_x = result_position_x
        self.output_position_y = result_position_y

        return None

    def backprop(self, error):
        #Result_shape = self.input_shape and error_shape = self.output_shape
        result = np.zeros(self.input.shape)

        for channel in range(self.output.shape[-1]):
            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    x_pos = self.output_position_x[i,j,channel]
                    y_pos = self.output_position_y[i,j,channel]
                    value = error[i,j,channel]

                    result[int(x_pos), int(y_pos), channel] = value
        
        return result

# Flatten
class Flatten:
    def __init__(self):
        self.input = []
        self.output = []

    def init_params(self, input_shape):
        return None

    def run(self):
        output_layer = self.input
        self.output = np.ravel(output_layer)
        return None

# Dense
class Dense:
    def __init__(self, class_num, bias=1, momentum=0.01, learning_rate=0.05):
        self.input = []
        self.output = []
        self.class_num = class_num
        self.weights = []
        self.bias = bias
        self.momentum = momentum
        self.learning_rate = learning_rate
        # Error calculation, backprop
        self.delta_weight = []
        self.error_calc_input = [] # assume it's like y_true, but it can be propagated to other layers
        self.prev_error = []
        self.passed_error = []
        self.error = []

    def init_params(self, input_shape):
        class_num = self.class_num

        # Validation
        if class_num <= 0 :
            raise Exception("Invalid class number")

        # Init weight
        input_size = 1
        for shape in input_shape:
            input_size *= shape
        # input_size = dense_input.size
        self.weights = np.random.uniform(-1, 1, (class_num, input_size + 1)) # +1 for bias
        
        self.reset_delta_weight()

        return None

    def reset_delta_weight(self):
        self.delta_weight = np.zeros(self.weights.shape)

    def calculate_error(self):
        # calculate derived error for layer weight
        # dNet/dweight = input
        derived_error_net = self.input.copy()

        self.error = np.matmul(
            self.prev_error.reshape(self.prev_error.size, 1),
            derived_error_net.reshape(1, derived_error_net.size),
        )

        # Calculate delta_weight with momentum
        # Assumption: delta_weight(n) = error + alpha * delta_weight(n-1)
        self.delta_weight = self.error + self.momentum * self.delta_weight

        # DEBUG
        print("self.weights_dense", self.weights, self.weights.shape)
        print("self.output_dense", self.output, self.output.shape)
        print("self.prev_error_dense", self.prev_error, self.prev_error.shape)
        print("derived_error_net_dense", derived_error_net, derived_error_net.shape)
        print("self.error_dense", self.error, self.error.shape)
        print("self.delta_weight_dense", self.delta_weight, self.delta_weight.shape)
        # DEBUG END

    def update_weight(self):
        # Assumption: weight(n) = weight(n-1) - lr * delta_weight(n-1)
        self.weight = self.weight - self.learning_rate * self.delta_weight

    def run(self):
        # Init variables
        self.input = np.append(self.input, self.bias) # Append bias to input. Assumption: input is already flattened
        class_num = self.class_num
        input_size = self.input.size
        self.output = np.zeros(class_num)

        # Calculate net output
        for c in range(class_num):
            for w in range(input_size):
                self.output[c] += self.input[w] * self.weights[c][w] 

        return None

    def backward(self):
        pass

def load_image(image_path):
    try:
        raw_img = skimage.io.imread(image_path)
    except:
        raise Exception("Failed to load image")

    return raw_img

