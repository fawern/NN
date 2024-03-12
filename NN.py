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
        
        # for i in range(len(self.layers)-1):
        #     output_shape = self.layers[i].num_neurons
        #     self.layers[i+1].set_weights(output_shape)
        
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
        
        layer1_output = self.layers[1].output
        layer2_output = self.layers[2].output
        layer3_output = self.layers[3].output
        layer4_output = self.layers[4].output

        # Output Layer
        error_output_layer = self.y - self.output
        delta_output_layer = error_output_layer * layer4_output * (1 - layer4_output)
        gradyan_weights_output = layer3_output.T.dot(delta_output_layer)    
        self.layers[4].weights += gradyan_weights_output * self.learning_rate

        # Hidden Layer
        error_hidden_layer_3 = delta_output_layer.dot(self.layers[4].weights.T)
        delta_hidden_layer_3 = error_hidden_layer_3 * layer3_output * (1 - layer3_output)
        gradyan_weights_3 = layer2_output.T.dot(delta_hidden_layer_3)
        self.layers[3].weights += gradyan_weights_3 * self.learning_rate

        error_hidden_layer_2 = delta_hidden_layer_3.dot(self.layers[3].weights.T)
        delta_hidden_layer_2 = error_hidden_layer_2 * layer2_output * (1 - layer2_output)
        gradyan_weights_2 = layer1_output.T.dot(delta_hidden_layer_2)
        self.layers[2].weights += gradyan_weights_2 * self.learning_rate

        # Input Layer
        erro_input_layer = delta_hidden_layer_2.dot(self.layers[2].weights.T)
        delta_input_layer = erro_input_layer * layer1_output * (1 - layer1_output)
        gradyan_weights_input = self.x.T.dot(delta_input_layer)
        self.layers[1].weights += gradyan_weights_input * self.learning_rate



        # error_output_layer = self.y - self.output
        # delta_output_layer = error_output_layer * layer3_output * (1 - layer3_output)
        # gradyan_weights_output = layer2_output.T.dot(delta_output_layer)
        # self.layers[3].weights += gradyan_weights_output * self.learning_rate

        # error_hidden_layer_2 = delta_output_layer.dot(self.layers[3].weights.T)
        # delta_hidden_layer_2 = error_hidden_layer_2 * layer2_output * (1 - layer2_output)
        # gradyan_weights_2 = layer1_output.T.dot(delta_hidden_layer_2)

        # self.layers[2].weights += gradyan_weights_2 * self.learning_rate

        # error_hidden_layer_1 = delta_hidden_layer_2.dot(self.layers[2].weights.T)
        # delta_hidden_layer_1 = error_hidden_layer_1 * layer1_output * (1 - layer1_output)
        # gradyan_weights_1 = self.x.T.dot(delta_hidden_layer_1)

        # self.layers[1].weights += gradyan_weights_1 * self.learning_rate

        # error_hidden_layer_4 = self.y - layer4_output
        # print('self.y.shape', self.y.shape)
        # print('layer4_output.shape', layer4_output.shape)
        # print('error_hidden_layer_4.shape', error_hidden_layer_4.shape)
        # delta_output_layer_4 = error_hidden_layer_4 * layer4_output * (1 - layer4_output)
        # print("delta_output_layer_4.shape", delta_output_layer_4.shape)
        # gradyan_weights_4 = layer3_output.T.dot(delta_output_layer_4)
        # print("gradyan_weights_4.shape", gradyan_weights_4.shape)
        # self.layers[4].weights += gradyan_weights_4 * self.learning_rate

        # error_hidden_layer_3 = delta_output_layer_4.dot(self.layers[4].weights.T)
        # delta_hidden_layer_3 = error_hidden_layer_3 * layer3_output * (1 - layer3_output)
        # gradyan_weights_3 = layer2_output.T.dot(delta_hidden_layer_3)
        # self.layers[3].weights += gradyan_weights_3 * self.learning_rate

        # error_hidden_layer_2 = delta_hidden_layer_3.dot(self.layers[3].weights.T)
        # delta_hidden_layer_2 = error_hidden_layer_2 * layer2_output * (1 - layer2_output)
        # gradyan_weights_2 = layer1_output.T.dot(delta_hidden_layer_2)
        # self.layers[2].weights += gradyan_weights_2 * self.learning_rate
        
        # error_hidden_layer_1 = delta_hidden_layer_2.dot(self.layers[2].weights.T)
        # delta_hidden_layer_1 = error_hidden_layer_1 * layer1_output * (1 - layer1_output)
        # gradyan_weights_1 = self.x.T.dot(delta_hidden_layer_1)
        # self.layers[1].weights += gradyan_weights_1 * self.learning_rate

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