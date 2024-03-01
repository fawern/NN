import numpy as np 
# np.random.seed(0)

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
        Activation functions for the model.

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
        exp_x = np.exp(x - np.max(x)) 
        return exp_x / np.sum(exp_x, axis=0)

    def tanh(self, x):
        return np.tanh(x)

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
        Add a layer to the model.

        Args:
            - layer (NLayer): The layer to add the model.
        """
        self.layers.append(layer)

    def train_model(self, x, output_data=None):
        """
        Train the model.

        Args:
            - x (np.array): The input data.
            - output_data (np.array): The output data.
        """

        ## First Layer
        # output_shape = self.layers[1].num_neurons
        # self.output = self.layers[0].get_weights(output_shape)

        print("===============================================================")

        output_shape = self.layers[1].num_neurons        
        self.layers[0].set_weights(output_shape)

        self.output = self.layers[0].forward(x)

        print(f"({1}/{len(self.layers)-1}) Layer {1} trained")

        print("---------------------------------------------------------------")

        ## Other Layers 1 2 3 4 , len=5
        for i in range(1, len(self.layers)-1): 
            output_shape = self.layers[i+1].num_neurons
            self.layers[i].set_weights(output_shape)

            self.output = self.layers[i].forward(self.output)

            print(f"({i+1}/{len(self.layers)-1}) Layer {i+1} trained")
            print("---------------------------------------------------------------")

    def predict_input(self):
        return self.output

class NInput:
    """"
    # Input Layer.
    """
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
    
    def set_weights(self, output_shape):
        self.weights = np.random.uniform(-1, 1, size=(self.num_neurons, output_shape))


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
        self.weights = np.random.uniform(-1, 1, size=(self.num_neurons, output_shape))

    # def set_activation(self, activation):
    #     self.activation = activation

    def forward(self, input_data):
        """
        Feed-Forward for the layer.

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
        

