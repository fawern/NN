import numpy as np 
np.random.seed(0)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class ActivationFunctions:
    def __init__(self):
        self.activation_functions_dict = {
            'sigmoid': self.sigmoid,  
            'sigmoid_derivative': self.sigmoid_derivative,
            'softmax': self.softmax,  
            'softmax_derivative': self.softmax_derivative,
            'tanh': self.tanh,       
            'tanh_derivative': self.tanh_derivative,
            'relu': self.relu,
            'relu_derivative': self.relu_derivative,
            'linear': self.linear,
            'linear_derivative': self.linear_derivative,
            'leaky_relu': self.leaky_relu,
            'leaky_relu_derivative': self.leaky_relu_derivative       
        }

    def add_activation_function(self, function_name, function_formula):
        """
        # Add a custom activation function to the activations dictionary.

        Args:
            - function_name (str): The name of the activation function.
            - function_formula (formula): The formula of the activation function.
        """
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

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x): return x * (1 - x)

    def softmax(self, x): return np.exp(x - np.max(x, axis=0))  / np.sum(np.exp(x - np.max(x, axis=0)) , axis=0)
    def softmax_derivative(self, x): return x * (1 - x)

    def tanh(self, x): return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def tanh_derivative(self, x): return 1 - np.power(x, 2)

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return np.where(x <= 0, 0, 1)

    def linear(self, x): return x 
    def linear_derivative(self, x): return 1 
    
    def leaky_relu(self, x): return np.where(x > 0, x, x * 0.01) 
    def leaky_relu_derivative(self, x): return np.where(x > 0, 1, 0.01)

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

        if len(self.layers) >= 2:
            for i in range(len(self.layers)-1):
                output_shape = self.layers[i].num_neurons
                self.layers[i+1].set_weights(output_shape)
            
    def train_model(self, x, y, iterations=1, learning_rate=0.001):
        """
        # Train the model.

        Args:
            - x (np.array): The input data.
            - output_data (np.array): The output data.
        """
        self.learning_rate = learning_rate
        self.x = x
        self.y = y
        
        for iter_ in range(iterations):
            self.output = self.x
            for i in range(1, len(self.layers)):
                self.output = self.layers[i].forward(self.output)

            loss = np.mean(np.square(self.y-self.output))
            self.losses.append(loss)
            self.backpropagation(learning_rate)

    def backpropagation(self, learning_rate):
        """
        # Backpropagation algorithm to update weights.

        Args:
            - learning_rate (float): The learning rate for updating weights.
        """
        # Output Layer
        error_output_layer = self.y - self.output
        layer_activation_function = self.layers[-1].get_activation() + '_derivative'
        derivative_output_layer = ActivationFunctions().activation_functions(layer_activation_function, self.output)
        delta_output_layer = error_output_layer * derivative_output_layer
        gradyan_weights_output = self.layers[-2].output.T.dot(delta_output_layer)    
        self.layers[-1].weights += gradyan_weights_output * self.learning_rate
        
        # If there are at least 1 hidden layer
        if len(self.layers) >= 3:
            # Hidden Layers
            delta_hidden_layer = delta_output_layer
            for i in range(len(self.layers)-2, 1, -1):
                error_hidden_layer = delta_hidden_layer.dot(self.layers[i+1].weights.T)
                layer_activation_function = self.layers[i].get_activation() + '_derivative'
                derivative_hidden_layer = ActivationFunctions().activation_functions(layer_activation_function, self.layers[i].output)
                delta_hidden_layer = error_hidden_layer * derivative_hidden_layer
                gradyan_weights = self.layers[i-1].output.T.dot(delta_hidden_layer)
                self.layers[i].weights += gradyan_weights * self.learning_rate
        
        # If there is no hidden layer
        else:
            delta_hidden_layer = delta_output_layer

        # Input Layer
        erro_input_layer = delta_hidden_layer.dot(self.layers[2].weights.T)
        layer_activation_function = self.layers[1].get_activation() + '_derivative'
        derivative_hidden_layer = ActivationFunctions().activation_functions(layer_activation_function, self.layers[1].output)
        delta_input_layer = erro_input_layer * derivative_hidden_layer
        gradyan_weights_input = self.x.T.dot(delta_input_layer)
        self.layers[1].weights += gradyan_weights_input * self.learning_rate

    def evaluate_trained_model(self):
        '''
        # Evaluate the trained model.

        Retuns:
            - float: The accuracy of the model.
            - np.array: The confusion matrix of the model.
        '''
        predicted_values = [1 if x > 0.5 else 0 for x in self.output]
        true_values = self.y

        true_predicts = [1 if x == y else 0 for x, y in zip(predicted_values, true_values)]

        accuracy = sum(true_predicts) / len(true_values)

        return accuracy, confusion_matrix(true_values, predicted_values)

    def show_loss_graph(self):
        """
        # Show the loss graph of the model.
        """
        plt.figure(figsize=(5, 3), facecolor='#032527')
        plt.title('Losses', color='white')
        plt.gca().set_facecolor('#032527') 
        plt.xticks(color='#5ECD5A')
        plt.yticks(color='#5ECD5A')
        plt.plot(self.losses)

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
    def __init__(self, num_neurons, activation='linear', use_bias=True, function_name=None, function_formula=None):
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
    
    def set_activation(self, activation):
        """
        # Set the activation function for the layer. 

        Args:
            - activation (str): The activation function to use.
        """
        self.activation = activation
    
    def get_activation(self):
        """
        # Get the activation function of the layer. 

        Returns:
            - str: The activation function of the layer.
        """
        return self.activation

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