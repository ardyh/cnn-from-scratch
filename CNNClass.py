import numpy as np

class Sequential:
    def __init__(self, input_shape, learning_rate=0.5, momentum=0.1):
        self.input = []
        self.layers = [] 
        self.final_output = []
        self.input_shape = input_shape
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
                
            layer.run()
            prev_output = layer.output.copy()
        
        self.final_output = prev_output

    def backprop(self, y_instance, is_update):
        prev_error = []; 
        error_calc_output = []
        for idx, layer in enumerate(reversed(self.layers)):
            if idx == 0: # If last layer
                layer.error_calc_input = y_instance
                layer.calculate_delta_output()
            else:
                layer.prev_error = prev_error
                layer.calculate_error()
            
            prev_error = layer.passed_error

        if (is_update):
            for layer in self.layers:
                layer.update_weight()

    def train(self, X, y, epochs=50, batch_size=5):
        instance_size = len(X)
        
        # Initialize parameters for every layer
        for layer in self.layers:
            layer.init_params(self.input_shape)

        # iterate every epoch
        for epoch in list(range(epochs)):
            print(f"Epoch {epoch}")
            # iterate every instance
            for instance_idx, instance in enumerate(zip(X, y)):
                X_instance = instance[0]; y_instance = instance[1]
                
                # If last instance or multiple of batch, update params
                is_update = (instance_idx == instance_size) or (instance_idx % batch_size == 0)
                
                self.forwardprop(X_instance)
                self.backprop(y_instance, is_update)
        return None

    def predict(self, X):
        y_pred = []
        for img in X:
            self.forwardprop(img)
            y_pred.append(np.argmax(self.final_output))

        return y_pred
    
# Conv Layer
class Conv2D:
    def __init__(self, filter_number, filter_size_length, filter_size_width, padding_layer=0, padded_number=0, stride=1, learning_rate=0.1, momentum=0.01):
        self.input = []
        self.output = []
        
        self.filter = []
        self.delta_filter = []
        self.error_filter = []
        self.filter_number = filter_number
        self.filter_size_length = filter_size_length
        self.filter_size_width = filter_size_width
        self.padding_layer = padding_layer
        self.padded_number = padded_number
        self.stride = stride

        self.filter_bias = []
        self.delta_bias = []
        self.error_bias = []

        self.prev_error = []
        self.passed_error = []

        #Define update weight variables
        self.learning_rate = learning_rate
        self.momentum = momentum

    def init_params(self, input_shape):
        # initialize filter
        self.init_filter(input_shape)

        # initial bias with 0
        self.filter_bias = np.zeros(self.filter.shape[0])

        # initialize delta weight and bias with 0
        self.delta_filter = np.zeros(self.filter.shape)
        self.delta_bias = np.zeros(self.filter_bias.shape)

        # initialize delta weight and bias with 0
        self.error_filter = np.zeros(self.filter.shape)
        self.error_bias = np.zeros(self.filter_bias.shape)
    
    def reset_delta_weight(self):
        self.delta_weight = np.zeros(self.weights.shape)

    def init_filter(self, input_shape):
        # we assume image only 2D, so only channel validation is needed
        # set the lower and upper bound of randomized parameters to be intialized in filter
        LOWER_BOUND = -10
        UPPER_BOUND = 10

        convolution_filter = []
        filter_channel = input_shape[-1]

        convolution_filter = np.zeros((self.filter_number, self.filter_size_length, self.filter_size_width, filter_channel))
        for i in range(self.filter_number):
            convolution_filter[i, :, :, :] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(self.filter_size_length, self.filter_size_width, filter_channel))

        self.filter = convolution_filter

    def add_padding(self, input_matrix):
        if (len(input_matrix.shape) > 2):
            # initialize result matrix
            padded_result = np.zeros((input_matrix.shape[0] + self.padding_layer * 2, 
                                    input_matrix.shape[1] + self.padding_layer * 2,
                                    input_matrix.shape[-1]))

            for i in range(input_matrix.shape[-1]):
                current_channel = input_matrix[:, :, i]
                padded_current_channel = np.pad(current_channel, self.padding_layer, mode='constant', constant_values=self.padded_number)
                padded_result[:, :, i] = padded_current_channel
        else:
            padded_result = np.pad(input_matrix, self.padding_layer, mode='constant', constant_values=self.padded_number)

        return padded_result

    def convolution_calculation(self, input_matrix, conv_filter, stride_x, stride_y):
        #input_matrix shape = (mxn)

        # initialize result matrix
        output_shape = (((input_matrix.shape[0] - conv_filter.shape[0]) // stride_x + 1),
                        ((input_matrix.shape[1] - conv_filter.shape[1]) // stride_y + 1))
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

                input_width_idx += stride_y
                output_width_idx +=  1
            
            input_length_idx += stride_x
            output_length_idx += 1

        return output_matrix

    def run(self):
        # validation
        if (len(self.input.shape) < 2):
            raise Exception("Invalid input matrix")
        
        if (self.filter_number <= 0 or self.filter_size_length <= 0 or self.filter_size_width <= 0 or self.stride <= 0 or self.padding_layer < 0) :
            raise Exception("Error input parameter")

        # add padding to input
        self.input = self.add_padding(self.input)

        # feature map formula : (W - F + 2P) / S + 1
        feature_map_shape = (((self.input.shape[0] - self.filter.shape[1]) // self.stride + 1),
                            ((self.input.shape[1] - self.filter.shape[2]) // self.stride + 1),
                            self.filter_number)
        
        # initialize feature map output
        feature_map = np.zeros(feature_map_shape)

        for filter_num in range(self.filter.shape[0]):
            current_filter = self.filter[filter_num, :]

            if len(current_filter.shape) > 2:
                convolution_result = self.convolution_calculation(self.input[:, :, 0], current_filter[:, :, 0], self.stride, self.stride)
                for channel in range(1, current_filter.shape[-1]):
                    convolution_result = convolution_result + self.convolution_calculation(self.input[:, :, channel], current_filter[:, :, channel], self.stride, self.stride)
            else:
                convolution_result = self.convolution_calculation(self.input, current_filter, self.stride, self.stride)
            
            feature_map[:, :, filter_num] = convolution_result + self.filter_bias[filter_num]
        
        self.output = feature_map
        return None

    def calculate_error(self):
        # calculate bias error 
        error = self.prev_error.copy() # Line tambahan, kalo ntar error coba debug ini
        for i in range (error.shape[-1]):
            self.error_bias[i] = np.sum(error[:,:,i])
        
        # update delta bias
        for i in range(len(self.delta_bias)):
            self.delta_bias[i] = self.learning_rate * self.error_bias[i] + self.momentum * self.delta_bias[i]


        # calculate dE / dW (Error Weight)
        # assuming filter and input are symmetrical
        calculated_stride_x = (self.input.shape[0] - error.shape[0]) // (self.filter.shape[1] - 1)
        calculated_stride_y = (self.input.shape[1] - error.shape[0]) // (self.filter.shape[2] - 1)
        for channel in range (self.input.shape[-1]):
            for filter_num in range(error.shape[-1]):
                product_matrix = self.convolution_calculation(self.input[:,:,channel], error[:,:,filter_num], calculated_stride_x, calculated_stride_y)
                self.error_filter[filter_num,:,:,channel] = product_matrix
                
        # Update delta_filter
        for filter_num in range(self.delta_filter.shape[0]):
            for i in range (self.delta_filter.shape[1]):
                for j in range (self.delta_filter.shape[2]):
                    for k in range (self.delta_filter.shape[3]):
                        self.delta_filter[filter_num,i,j,k] = self.learning_rate * self.error_filter[filter_num,i,j,k] + self.momentum * self.delta_filter[filter_num,i,j,k]

    def update_weight(self):
        # bias update
        for i in range(len(self.delta_bias)):
            # self.delta_bias[i] = self.learning_rate * self.error_bias[i] + self.momentum * self.delta_bias[i]
            self.filter_bias[i] -= self.delta_bias[i]  

        # weight update
        for filter_num in range(self.delta_filter.shape[0]):
            for i in range (self.delta_filter.shape[1]):
                for j in range (self.delta_filter.shape[2]):
                    for k in range (self.delta_filter.shape[3]):
                        # self.delta_filter[filter_num,i,j,k] = self.learning_rate * self.error_filter[filter_num,i,j,k] + self.momentum * self.delta_filter[filter_num,i,j,k]
                        self.filter[filter_num,i,j,k] -= self.delta_filter[filter_num,i,j,k]

        self.delta_filter = np.zeros(self.filter.shape)
        self.delta_bias = np.zeros(self.filter_bias.shape)

# Activation
class Activation:
    def __init__(self, function_name="relu", class_num=None):
        self.input = []
        self.output = []
        self.function_name = function_name
        self.class_num = class_num
        # Error calculation, backprop
        self.error_calc_input = []
        self.prev_error = []
        self.passed_error = []
        self.error = [] 

    def init_params(self, input_shape):
        return None

    def calculate_delta_output(self):
        # Assuming untuk output layer
        true_val = [1 if i == self.error_calc_input else 0 for i in range(self.class_num)]
        class_errors = []
        for y_true_val, output_val in zip(true_val, self.output):
            
            class_error_term = 0
            if (self.function_name == "sigmoid"):
                class_error_term = (y_true_val - output_val) * output_val * (1 - output_val) 
            elif (self.function_name == "relu"):
                relu_derivative_val = 1 if output_val > 0 else 0
                class_error_term = (y_true_val - output_val) * relu_derivative_val
            else:
                raise Exception("Invalid activation function name")
        
            class_errors.append(class_error_term)
        
        self.passed_error = np.array(class_errors)
    
    # Assumption only works for layers where output dimension of next layer is equal to activation layer's input dimension
    def calculate_error(self):
        result = np.zeros(self.output.shape)

        for channel in range(self.output.shape[-1]):
            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    if(self.output[i,j,channel] > 0):
                        result[i,j,channel] = self.prev_error[i,j,channel]
                    else:
                        result[i,j,channel] = 0
        
        self.passed_error = result.copy()
        
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
        function_name = self.function_name

        # activation function defaults to relu
        if (function_name == "sigmoid"):
            v_activation = np.vectorize(self.sigmoid)
        elif (function_name == "relu"):
            v_activation = np.vectorize(self.relu)
        else:
            raise Exception("Invalid activation function name")

        self.output = v_activation(self.input)
        return None

    def backprop(self, error):
        result = np.zeros(self.output.shape)

        for channel in range(self.output.shape[-1]):
            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    if(self.output[i,j,channel] > 0):
                        result[i,j,channel] = error[i,j,channel]
                    else:
                        result[i,j,channel] = 0
        
        return result

    def update_weight(self):
        #No weight to be updated in detection stage
        return None

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
        # Error calculation, backprop
        self.prev_error = []
        self.passed_error = []

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

    def calculate_error(self):
        error = self.prev_error.copy()
        
        # Result_shape = self.input_shape and error_shape = self.output_shape
        result = np.zeros(self.input.shape)

        for channel in range(self.output.shape[-1]):

            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    x_pos = self.output_position_x[i,j,channel]
                    y_pos = self.output_position_y[i,j,channel]
                    value = error[i,j,channel]

                    result[int(x_pos), int(y_pos), channel] = value
                
        self.passed_error = result.copy()   


    def update_weight(self):
        # No weight to be updated in pooling stage
        return None

# Flatten
class Flatten:
    def __init__(self):
        self.input = []
        self.output = []
        # Error calculation, backprop
        self.prev_error = []
        self.passed_error = []

    def init_params(self, input_shape):
        return None

    def calculate_error(self):
        error = self.prev_error.copy() # excluding bias from prev error
        self.passed_error = error.reshape(self.input.shape)

    def update_weight(self):
        return None

    def run(self):
        output_layer = self.input.copy()
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

        return None

    def reset_delta_weight(self):
        self.delta_weight = np.zeros(self.weights.shape)

    def calculate_error(self):
        # Error to be passed to previous layer
        # dNet/dRelu = weight
        # dError/dRelu
        self.passed_error = np.matmul(
            self.prev_error.reshape(1, self.prev_error.size),
            self.weights
        )[:, :-1] # excluding bias

        # calculate derived error for layer weight
        # dNet/dWeight = input
        # dError/dWeight
        self.error = np.matmul(
            self.prev_error.reshape(self.prev_error.size, 1),
            self.input.reshape(1, self.input.size),
        )

        # Calculate delta_weight with momentum
        # Assumption: delta_weight(n) = error + alpha * delta_weight(n-1)
        self.delta_weight = self.error + self.momentum * self.delta_weight

    def update_weight(self):
        # Assumption: weight(n) = weight(n-1) - lr * delta_weight(n-1)
        self.weights = self.weights - self.learning_rate * self.delta_weight
        self.reset_delta_weight()

    def run(self):
        # Init variables
        self.input = np.append(self.input, self.bias) # Append bias to input. Assumption: input is already flattened
        
        # Init weight if first forwardprop
        if self.weights == []:
            self.weights = np.random.uniform(-1, 1, (self.class_num, self.input.size))
            self.reset_delta_weight()

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

