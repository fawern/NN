import numpy as np 

def activation_functions(activation, x):
    """
    Activation functions for the model.

    Args:
        - activation (str): The activation function to use.
        - x (np.array): The input data.
    
    Returns:
        - np.array: The output data after applying the activation function.
    """
    if activation == 'softmax':
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))

    else:
        raise ValueError(f"{activation_func} is not a valid activation function!!!")


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
    def __init__(self, shapes, activation, use_bias=True):
        self.shapes = shapes
        self.activation = activation
        self.use_bias= use_bias

        self.weights = np.random.uniform(-1, 1, size=self.shapes)
        self.bias = np.random.uniform(-1, 1)

    def forward(self, input_data):
        """
        Feed-Forward for the layer.

        Args:
            - input_data (np.array): The input data to the layer.
        """
        self.output = np.dot(input_data, self.weights)
        if self.use_bias: 
            self.output += self.bias

        self.output = activation_functions(self.activation, self.output)

        return self.output