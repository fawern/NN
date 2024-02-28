import numpy as np 

class ActivationFunctions:
    def __init__(self):
        self.activation_functions_dict = {}
        
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

        try:
            if activation == 'sigmoid': return 1 / (1 + np.exp(-x))
            elif activation == 'softmax': return np.exp(x) / np.sum(np.exp(x), axis=0)
            elif activation == 'tanh': return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            elif activation == 'relu': return x if x > 0 else 0

            # self.activation_functions_dict = {
            #         "sigmoid" : 1 / (1 + np.exp(-x)), 
            #         "softmax" : np.exp(x) / np.sum(np.exp(x), axis=0),
            #         'tanh' : (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
            #         'relu' : x if x > 0 else 0
            #     }
        
            # return self.activation_functions_dict[activation]

        except:
            raise ValueError(f"{activation} is not a valid activation function!!!")


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
        self.output = self.layers[0].forward(x)

        print("===============================================================")
        print(f"({1}/{len(self.layers)}) Layer {1} trained")
        print("---------------------------------------------------------------")

        ## Other Layers
        for i in range(1, len(self.layers)): 
            self.output = self.layers[i].forward(self.output)
            print(f"({i+2}/{len(self.layers)}) Layer {i+2} trained")
            print("---------------------------------------------------------------")
    
    def predict_input(self):
        return self.output

class NInput:
    """"
    # Input Layer.
    """
    def __init__(self, input_shapes):
        self.input_shapes = input_shapes


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
    def __init__(self, num_neurons, output_shape, activation, use_bias=True, function_name=None, function_formula=None):
        self.num_neurons = num_neurons
        self.output_shape = output_shape
        self.activation = activation
        self.use_bias= use_bias
        self.function_name = function_name
        self.function_formula = function_formula

        self.weights = np.random.uniform(-1, 1, size=(self.num_neurons, self.output_shape))
        self.bias = np.random.uniform(-1, 1)

    def forward(self, input_data):
        """
        Feed-Forward for the layer.

        Args:
            - input_data (np.array): The input data to the layer.
        """
        activation_function = ActivationFunctions()
        
        self.output = np.dot(input_data, self.weights)

        if self.use_bias: 
            self.output += self.bias
            
        self.output = activation_function.activation_functions(self.activation, self.output)
        return self.output