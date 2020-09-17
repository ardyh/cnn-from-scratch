from random import seed
from random import random
from math import exp

# External dependencies, make sure it's installed
import pandas as pd
import numpy as np
from PIL import Image

# Update Ardy: 13/09/2020
# Masih Implementasi MLP buat tugas ML
# Using Batch Stochastic Gradient Descent
# Baru add method process_image() buat dapet image dan process jadi np.array
class CNN:
    # constructor
    def __init__(self, n_inputs, n_hidden, n_outputs=3, bias=1, learning_rate=0.1):
        # LEGACY MLP ATTRIBUTES
        self.n_inputs = n_inputs  # number of input unit
        self.n_hidden = n_hidden  # number of hidden unit
        self.n_outputs = n_outputs  # number of output unit
        self.bias = bias  # bias parameter
        self.learning_rate = learning_rate

        # parameters of weight on input to hidden layer
        self.weights_ItoH = np.random.uniform(-1, 1, (n_inputs+1, n_hidden))
        self.dweights_ItoH = np.zeros((n_inputs+1, n_hidden))

        # parameters of weight on hidden to output layer
        self.weights_HtoO = np.random.uniform(-1, 1, (n_hidden+1, n_outputs))
        self.dweights_HtoO = np.zeros((n_hidden+1, n_outputs))

        # output value and error of hidden layer
        self.pre_activation_H = np.zeros(n_hidden)
        self.post_activation_H = np.zeros(n_hidden)
        self.error_H = np.zeros(n_hidden)

        # output value and error of output layer
        self.pre_activation_O = np.zeros(n_outputs)
        self.post_activation_O = np.zeros(n_outputs)
        self.error_O = np.zeros(n_outputs)
    
        # NEWLY MADE CNN ATTRIBUTES
        self.input = []
        

    # NEWLY MADE CNN METHODS
    def process_image(self, image_path):
        img = Image.open(image_path, 'r')
        img = np.array(list(img.getdata()))
        self.input = np.transpose(img)

    # LEGACY MLP METHODS
    # Net calculation method
    # Calculate net for an input
    def calculate_net_ItoH(self, sample, node):
        input_plus_bias = np.append(self.data[sample, :], self.bias)
        return np.dot(input_plus_bias, self.weights_ItoH[:, node])
    # Calculate net for a hidden unit

    def calculate_net_HtoO(self, node):
        hidden_plus_bias = np.append(self.post_activation_H, self.bias)
        return np.dot(hidden_plus_bias, self.weights_HtoO[:, node])

    # activation function
    def activation(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def one_hot_encode(self, target):
        encoder = OneHotEncoder(sparse=False)
        new_target = target.reshape(len(target), 1)
        target_encode = encoder.fit_transform(new_target)
        return target_encode

    # fit the network to the data
    def fit(self, data, target, epoch_limit=100, mini_batch_limit=10):
        self.data = data
        self.target = self.one_hot_encode(target)
        self.epoch_limit = epoch_limit

        len_data = len(data)

        # iterate each epoch
        for epoch in range(epoch_limit):

            # iterate each instance
            mini_batch_count = 0
            for instance in range(len_data):

                # From input layer to hidden layer
                # iterate every hidden layer to fill the values
                for hidden_unit in range(self.n_hidden):
                    # calculate the net input
                    self.pre_activation_H[hidden_unit] = self.calculate_net_ItoH(
                        instance, hidden_unit)
                    # calculate the activated value
                    self.post_activation_H[hidden_unit] = self.activation(
                        self.pre_activation_H[hidden_unit])

                # From hidden layer to output layer
                for output_unit in range(self.n_outputs):
                    # calculate the net input
                    self.pre_activation_O[output_unit] = self.calculate_net_HtoO(
                        output_unit)
                    # calculate the activated value
                    self.post_activation_O[output_unit] = self.activation(
                        self.pre_activation_O[output_unit])

                # Backpropagation
                # if already at minibatch limit or at the last instance, update the weight
                if((mini_batch_count == mini_batch_limit) or (instance == len_data - 1)):

                    # update weight - input to hidden
                    self.weights_ItoH = np.add(
                        self.weights_ItoH, self.dweights_ItoH)
                    # update weight - hidden to output
                    self.weights_HtoO = np.add(
                        self.weights_HtoO, self.dweights_HtoO)

                    # reset delta weight to zero
                    self.dweights_ItoH = np.zeros(
                        (self.n_inputs+1, self.n_hidden))
                    self.dweights_HtoO = np.zeros(
                        (self.n_hidden+1, self.n_outputs))

                    # reset iterator
                    mini_batch_count = 0

                # if below minibatch limit, update delta-weight
                else:
                    # update delta-weight from output
                    # (+1 accomodating bias)
                    for hidden_unit in range(self.n_hidden + 1):
                        for output_unit in range(self.n_outputs):
                            # (Minus sign merged). Formula: (target_ok - out_ok) * out_ok * (1 - out_ok) * out_hj
                            target_o = self.target[instance][output_unit]
                            out_o = self.post_activation_O[output_unit]

                            # calculating weight of bias
                            if (hidden_unit == self.n_hidden):
                                out_h = self.bias
                            # calculating weight of activated hidden unit
                            else:
                                out_h = self.post_activation_H[hidden_unit]

                            self.error_O[output_unit] = (
                                target_o - out_o) * out_o * (1 - out_o)
                            self.dweights_HtoO[hidden_unit][output_unit] += self.error_O[output_unit] * \
                                out_h * self.learning_rate

                    # update delta-weight from hidden layer
                    # (+1 accomodating bias)
                    for input_unit in range(self.n_inputs + 1):
                        for hidden_unit in range(self.n_hidden):
                            #### Formula: sigma_ok(error_o * w_ho) * out_hj * (1 - out_hj) * input_i
                            sigma_err_output = np.dot(
                                self.error_O, self.weights_HtoO[hidden_unit, :])
                            out_h = self.post_activation_H[hidden_unit]

                            # calculating weight of bias
                            if(input_unit == self.n_inputs):
                                input_i = self.bias
                            # calculating weight of input unit
                            else:
                                input_i = self.data[instance, input_unit]

                            self.error_H[hidden_unit] = sigma_err_output * \
                                out_h * (1 - out_h)
                            self.dweights_ItoH[input_unit][hidden_unit] += self.error_H[hidden_unit] * \
                                input_i * self.learning_rate

                    # increment iterator
                    mini_batch_count += 1

    def predict(self, data):
        self.data = data
        result = []
        # iterate each instance
        for instance in range(len(data)):
            # iterate every hidden layer to fill the values
            for hidden_unit in range(self.n_hidden):
                # calculate the net input
                self.pre_activation_H[hidden_unit] = self.calculate_net_ItoH(
                    instance, hidden_unit)
                # calculate the activated value
                self.post_activation_H[hidden_unit] = self.activation(
                    self.pre_activation_H[hidden_unit])

            max_value = 0
            max_index = -1
            # From hidden layer to output layer
            for output_unit in range(self.n_outputs):
                # calculate the net input
                self.pre_activation_O[output_unit] = self.calculate_net_HtoO(
                    output_unit)
                # calculate the activated value
                self.post_activation_O[output_unit] = self.activation(
                    self.pre_activation_O[output_unit])
                if(self.post_activation_O[output_unit] >= max_value):
                    max_value = self.post_activation_O[output_unit]
                    max_index = output_unit

            result = np.append(result, max_index)

        return result

    # print weight
    def print_w_ItoH(self):
        index = []
        for n in range(self.n_inputs+1):
            index.append('WInput'+str(n))
        column = []
        for n in range(self.n_hidden):
            column.append('Hidden'+str(n))
        print(pd.DataFrame(net.weights_ItoH, index, column))

    # print weight
    def print_w_HtoO(self):
        index = []
        for n in range(self.n_hidden+1):
            index.append('WHidden'+str(n))
        column = []
        for n in range(self.n_outputs):
            column.append('Output'+str(n))
        print(pd.DataFrame(net.weights_HtoO, index, column))


