import numpy as np 
np.random.seed(0)

class ActivationFunctions:
    def __init__(self):
        self.activation_functions_dict = {
            'sigmoid': self.sigmoid,  
            'softmax': self.softmax,  
            'tanh': self.tanh,       
            'relu': self.relu         
        }

    def add_activation_function(self, function_name, function_formula):
        self.activation_functions_dict[function_name] = function_formula

    def activation_functions(self, activation, x):
        """
        # Activation functions for the model.

        Args:
            - activation (str): The activation function to use.
            - x (np.array): The input data.

        Returns:
            - np.array: The output data after applying the activation function.
        """
        
        if activation in self.activation_functions_dict:
            return self.activation_functions_dict[activation](x)
        else:
            raise ValueError(f"{activation} is not a valid activation function!!!")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0)) 
        return exp_x / np.sum(exp_x, axis=0)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

class Layers:
    """
    # Simple Multi-Layer-Perceptron Model.

    Args:
        - layers (list): The list of layers to add to the model.

    Implementation:
        - The model is trained by calling the train_model method and passing the input data.
        - The model is used to predict the output by calling the predict_input method.
    """
    def __init__(self):
        self.layers = []

    def add(self, layer):
        """
        # Add a layer to the model.

        Args:
            - layer (NLayer): The layer to add the model.
        """
        self.layers.append(layer)

    def train_model(self, x, y, iterations=1, learning_rate=0.001):
        """
        # Train the model.

        Args:
            - x (np.array): The input data.
            - output_data (np.array): The output data.
        """

        self.x = x
        self.y = y

        print("=============================== Iteration 1 ===============================")
        for iter_ in range(iterations):
            output_shape = self.layers[0].num_neurons        
            self.layers[1].set_weights(output_shape)
            
            self.output = self.layers[1].forward(self.x)

            print(f"({1}/{len(self.layers)-1}) Layer {1} trained")

            for i in range(2, len(self.layers)): 
                output_shape = self.layers[i-1].num_neurons
                self.layers[i].set_weights(output_shape)

                self.output = self.layers[i].forward(self.output)

                print(f"({i}/{len(self.layers)-1}) Layer {i} trained")
            print(f"=============================== Iteration {iter_ + 2} ===============================")

            self.backpropagation(learning_rate)

    def backpropagation(self, learning_rate):
        """
        # Backpropagation algorithm to update weights.

        Args:
            - learning_rate (float): The learning rate for updating weights.
        """
        output_error = self.output - self.y.reshape(-1, 1)
        
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev_layer_output = self.layers[i-1].output if i > 1 else self.x

            grad_weights = np.dot(prev_layer_output.T, output_error * layer.output * (1 - layer.output))
            
            layer.weights -= learning_rate * grad_weights
            
            if layer.use_bias:
                grad_bias = np.sum(output_error * layer.output * (1 - layer.output), axis=0)
                layer.bias -= learning_rate * grad_bias
                
            output_error = np.dot(output_error * layer.output * (1 - layer.output), layer.weights.T)

    def evaluate_trained_model(self):
        predicted_values = [1 if x > 0.5 else 0 for x in self.output]
        true_values = self.y

        true_predicts = [1 if x == y else 0 for x, y in zip(predicted_values, true_values)]

        accuracy = sum(true_predicts) / len(true_values)

        return accuracy

    def predict_input(self):
        return self.output

class NInput:
    """
    # Input Layer.
    """
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

class NLayer:
    """
    # Simple Multi-Layer-Perceptron Layer.

    Args:
        - shapes (tuple): The shape of the input and output of the layer.
        - activation (str): The activation function to use.
        - use_bias (bool): Whether to use bias in the layer or not, default is True. 

    Implementation: 
        - weights are initialized with random values between -1 and 1.and
        - bias is initialized with random value between -1 and 1. 
    """
    def __init__(self, num_neurons, activation=None, use_bias=True, function_name=None, function_formula=None):
        self.num_neurons = num_neurons
        self.activation = activation
        self.use_bias= use_bias
        self.function_name = function_name
        self.function_formula = function_formula

        self.bias = np.random.uniform(-1, 1)
    
    def set_weights(self, output_shape):
        self.weights = np.random.uniform(-1, 1, size=(output_shape, self.num_neurons))

    def forward(self, input_data):
        """
        # Feed-Forward for the layer.

        Args:
            - input_data (np.array): The input data to the layer.
        """
        self.output = np.dot(input_data, self.weights)

        if self.use_bias: 
            self.output += self.bias

        if self.activation is not None:
            activation_function = ActivationFunctions()    
            self.output = activation_function.activation_functions(self.activation, self.output)
            return self.output

        else:
            return self.output