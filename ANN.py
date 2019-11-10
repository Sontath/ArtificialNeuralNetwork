# -*- coding: utf-8 -*-
"""
@author: Bradley
"""
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Node:
    def __init__(self):
        self.output = 1
        self.weights = []
        self.delta = 1
    
    def __str__(self):
        weights = ""
        for weight in self.weights:
            weights += str(weight) + ", "
        return "\tOutput: {:.4f}, Delta: {:.4f}\n\tWeights: {}\n".format(self.output, self.delta, weights)
    
    def node_constructor(self, size, is_input, value = None):
        self.is_input = is_input
        
        if value == None:
            self.output = random.random()
        else:
            self.output = value
            
        for x in range(size):
            self.weights.append(random.randrange(-1, 1))
            
        self.delta = random.random()
        return self

class Layer:
    def __init__(self):
        self.nodes = []
        self.bias = Node()
        
    
    def __str__(self):
        value = "Layer:\n"
        
        for node in self.nodes:
            value += str(node)
        
        return value
    
    
    def create_layer(self, size, weight_size, is_input, bias):
        self.bias = (Node.node_constructor(Node(), weight_size, is_input, bias))
        
        for x in range(size):
            self.nodes.append(Node.node_constructor(Node(), weight_size, is_input))
        
        return self

class Artificial_Neural_Network:
    def __init__(self):
        self.input_count = 1
        self.layer_count = 1
        self.layer_depth_count = 1
        self.output_count = 1
        self.bias = 1
        self.layers = []
        self.learning_rate = 1
    
    def __str__(self):
        value = "Network is:\n"
        
        for x in self.layers:
            value += str(x)
        
        return value
    
    def create_network(self):
        self.layers.append(Layer.create_layer(Layer(), self.input_count, self.layer_depth_count, True, self.bias))
        
        for x in range(self.layer_count - 1):
            self.layers.append(Layer.create_layer(Layer(), self.layer_depth_count, self.layer_depth_count, False, self.bias))
        
        self.layers.append(Layer.create_layer(Layer(), self.layer_depth_count, self.output_count, False, self.bias))
        self.layers.append(Layer.create_layer(Layer(), self.output_count, 0, False, self.bias))
    
    def set_inputs(self, array):
        for x, y in zip(array, self.layers[0].nodes):
            y.output = x
    
    def get_outputs(self):
        array = []
        for node in self.layers[len(self.layers) - 1].nodes:
            array.append(node.output)
        return array
    
    def calc_output(self):
        buffer_val = 0
        # for each col that we need to calculate output
        
        # For each node in the previous row, calculate the weight added from that node.
        # put that sum into the sigmoid function of (1 / (1 + e^(-(sum))))
        for col_number in range(1, len(self.layers)):
            for current_node in range(len(self.layers[col_number].nodes)):
                buffer_val = 0
                for prev_node in self.layers[col_number - 1].nodes:
                    # add weighted sum from prev_node
                    buffer_val += (prev_node.output * prev_node.weights[current_node])
                # add weighted bias from previos layer
                # calculate sigmoid function and store in output
                buffer_val += (self.layers[col_number - 1].bias.output * self.layers[col_number - 1].bias.weights[current_node])
                buffer_val = (1 / (1 + math.pow(math.e, (-buffer_val))))
                self.layers[col_number].nodes[current_node].output = buffer_val
    
    # Need to make a back propogation function to make the ANN "learn" by updating the weights
    def learn(self, target):        
        buffer_value = 0
  
        # make sure the output values are accurate to current weights
        self.calc_output()
        
        # Starting from the output to the inputs calculate delta
        for col_number in range(len(self.layers) - 1, -1, -1):
            # if this is an output node
            if col_number == len(self.layers) - 1:
                #output delta updates
                for node_count in range(len(self.layers[col_number].nodes)):
                    self.layers[col_number].nodes[node_count].delta = self.layers[col_number].nodes[node_count].output * ( 1 - self.layers[col_number].nodes[node_count].output) * (self.layers[col_number].nodes[node_count].output - target[node_count])
                    #DEBUGprint("Output bias: {}".format(self.layers[col_number].nodes[node_count].delta))
            else:
                #calculate for bias delta
                buffer_value = 0
                for next_node_count in range(len(self.layers[col_number + 1].nodes)):
                    buffer_value += (self.layers[col_number + 1].nodes[next_node_count].delta * self.layers[col_number].bias.weights[next_node_count])
                self.layers[col_number].bias.delta = self.layers[col_number].bias.output * ( 1 - self.layers[col_number].bias.output) * buffer_value
                # DEBUGprint("Bias delta: {}".format(self.layers[col_number].bias.delta))
                #calculate for node delta
                for node_count in range(len(self.layers[col_number].nodes)):
                    buffer_value = 0
                    for next_node_count in range(len(self.layers[col_number + 1].nodes)):
                        buffer_value += (self.layers[col_number + 1].nodes[next_node_count].delta * self.layers[col_number].nodes[node_count].weights[next_node_count])
                    self.layers[col_number].nodes[node_count].delta = self.layers[col_number].nodes[node_count].output * ( 1 - self.layers[col_number].nodes[node_count].output) * buffer_value
                    #DEBUGprint("Node bias: {}".format(self.layers[col_number].nodes[node_count].delta))
        # end of for loop
        
        
        # update weights using the current deltas for each node
        for layer in range(len(self.layers) - 1):
            for weight in range(len(self.layers[layer].bias.weights)):
                self.layers[layer].bias.weights[weight] = self.layers[layer].bias.weights[weight] - (-1 * self.learning_rate * self.layers[layer + 1].nodes[weight].delta * self.layers[layer].bias.output)
            for node in range(len(self.layers[layer].nodes)):
                for weight in range(len(self.layers[layer].nodes[node].weights)):
                    self.layers[layer].nodes[node].weights[weight] = self.layers[layer].nodes[node].weights[weight] - (-1 * self.learning_rate * self.layers[layer + 1].nodes[weight].delta * self.layers[layer].nodes[node].output) # value in parenthesis is the change in weight

# for Iris dataset
network = Artificial_Neural_Network()
network.input_count = 4
network.layer_count = 1
network.layer_depth_count = 2
network.output_count = 3
network.bias = -1
network.learning_rate = .01

network.create_network()

iris = open("iris.data", "r")

inputs = []
outputs = []
for line in iris:
    split_line = line.split(",")
    if len(split_line) == 5:
        input_buffer = []
        output_buffer = []
        
        for number in split_line[:4]:
            input_buffer.append(float(number))
        inputs.append(input_buffer)
        
        if split_line[4] == "Iris-setosa\n":
            output_buffer = [1, 0, 0]
        elif split_line[4] == "Iris-versicolor\n":
            output_buffer = [0, 1, 0]
        else:
            output_buffer = [0, 0, 1]
        
        outputs.append(output_buffer)

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = .33)

count = 0
for case, target in zip(x_train, y_train):
    network.set_inputs(case)
    network.learn(target)

count = 0
accuracy_count = 0
plot_count = [0, 0, 0, 0, 0, 0]
plot_categories = ["SetTP", "SetFP", "VerTP", "VerFP", "VirTP", "VirFP"]
last_target = []
for test, target in zip(x_test, y_test):
    count += 1
    network.set_inputs(test)
    network.calc_output()
    predicted_output = network.get_outputs()
    
    predicted_array = []
    
    if predicted_output[0] > predicted_output[1] and predicted_output[0] > predicted_output[2]:
        predicted_array = [1, 0, 0]
    elif predicted_output[1] > predicted_output[0] and predicted_output[1] > predicted_output[2]:
        predicted_array = [0, 1, 0]
    elif predicted_output[2] > predicted_output[1] and predicted_output[2] > predicted_output[1]:
        predicted_array = [0, 0, 1]
    
    if predicted_array == target:
        accuracy_count += 1
        if predicted_array[0] == 1:
            plot_count[0] += 1
        elif predicted_array[1] == 1:
            plot_count[2] += 1
        else:
            plot_count[4] += 1
    elif len(predicted_array) > 1:
        if predicted_array[0] == 1:
            plot_count[1] += 1
        elif predicted_array[1] == 1:
            plot_count[3] += 1
        else:
            plot_count[5] += 1
    
    last_target = target
        

print("Accuracy on Iris dataset was {:.1f}%".format((accuracy_count / len(x_test)) * 100))
count = 0
colors = {0: '#add8e6', 1: '#00008b', 2: '#ffcccb', 3: '#8b0000', 4:'#90ee90', 5: '#013220'}
for x, y in zip(plot_categories, plot_count):
    plt.bar(x, y, label=x, color=colors[count])
    count += 1

plt.xlabel("Categories")
plt.ylabel("Count")
plt.title("Accuracy of ANN")
plt.legend()
plt.show()
