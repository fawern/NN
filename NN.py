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
        self.losses = []

    def train_model(self, x, y, iterations=1, learning_rate=0.1):
        """
        # Train the model.

        Args:
            - x (np.array): The input data.
            - output_data (np.array): The output data.
        """

        self.x = x
        self.y = y

        # for iter_ in range(iterations):
        #     output_shape = self.layers[0].num_neurons        
        #     self.layers[1].set_weights(output_shape)
            
        #     self.output = self.layers[1].forward(self.x)

        #     for i in range(2, len(self.layers)): 
        #         output_shape = self.layers[i-1].num_neurons
        #         self.layers[i].set_weights(output_shape)
    
        #         self.output = self.layers[i].forward(self.output)
        
        output_shape = self.layers[0].num_neurons
        self.layers[1].set_weights(output_shape)

        output_shape = self.layers[1].num_neurons
        self.layers[2].set_weights(output_shape)

        output_shape = self.layers[2].num_neurons
        self.layers[3].set_weights(output_shape)

        for iter_ in range(iterations):
            self.output = self.layers[1].forward(self.x)
            self.output = self.layers[2].forward(self.output)
            self.output = self.layers[3].forward(self.output)

            loss = np.mean(np.square(self.y-self.output))
            self.losses.append(loss)

            self.backpropagation(learning_rate)

    def backpropagation(self, learning_rate):
        """
        # Backpropagation algorithm to update weights.

        Args:
            - learning_rate (float): The learning rate for updating weights.
        """
        layer1_output = self.layers[1].output
        layer2_output = self.layers[2].output
        output = self.output

        error_hidden_layer_3 = self.y - output
        delta_output_layer = error_hidden_layer_3 * output * (1 - output)
        
        error_hidden_layer_2 = delta_output_layer.dot(self.layers[3].weights.T)
        delta_hidden_layer_2 = error_hidden_layer_2 * layer2_output * (1 - layer2_output)

        error_hidden_layer_1 = delta_hidden_layer_2.dot(self.layers[2].weights.T)
        delta_hidden_layer_1 = error_hidden_layer_1 * layer1_output * (1 - layer1_output)
        
        self.layers[3].weights += layer2_output.T.dot(delta_output_layer) * learning_rate
        self.layers[2].weights += layer1_output.T.dot(delta_hidden_layer_2) * learning_rate
        self.layers[1].weights += self.x.T.dot(delta_hidden_layer_1) * learning_rate

    # def evaluate_trained_model(self):
    #     predicted_values = [1 if x > 0.5 else 0 for x in self.output]
    #     true_values = self.y

    #     true_predicts = [1 if x == y else 0 for x, y in zip(predicted_values, true_values)]

    #     accuracy = sum(true_predicts) / len(true_values)

    #     return accuracy

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
    
    def set_weights(self, output_shape=None, new_weights=None):
        """
        # Set the weights for the layer. 

        Args:
            - output_shape (int): The output shape of the layer.
            - new_weights (np.array): The new weights to set for the layer.

        Implementation:
            - If both output_shape and new_weights are None, then an error is raised.
            - If output_shape is not None, then the weights are initialized with random values between -1 and 1.
            - If new_weights is not None, then the weights are set to the new_weights
        """

        if output_shape is None and new_weights is None:
            raise ValueError("Both output_shape and new_weights cannot be None!!!")

        elif output_shape is not None:
            self.weights = np.random.uniform(-1, 1, size=(output_shape, self.num_neurons))

        elif new_weights is not None:
            self.weights = new_weights

        elif output_shape is not None and new_weights is not None:
            raise ValueError("One of output_shape and new_weights must be None!!!")

    def get_weights(self):
        '''
        # Get the weights of the layer.

        Returns:
            - np.array: The weights of the layer.
        '''
        return self.weights

    def forward(self, input_data):
        """
        # Feed-Forward for the layer.

        Args:
            - input_data (np.array): The input data to the layer.
        """
        self.output = np.dot(input_data, self.get_weights())

        if self.use_bias: 
            self.output += self.bias

        if self.activation is not None:
            activation_function = ActivationFunctions()    
            self.output = activation_function.activation_functions(self.activation, self.output)
            return self.output

        else:
            return self.output