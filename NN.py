import numpy as np 


def activation_functions(activation, x):

    if activation == 'softmax':
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(f"{activation_func} is not a valid activation function!!!")

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(x, y):
        output = self.layers[0].forward(x)

        for i in range(1, len(self.layers)):
            output = self.layers[1].forward(output)
            output = self.layers[1].forward(output)

        for layer in self.layers[2:]:
            output = layer_1.forward(inputs)

class Dense:
    def __init__(self, shapes, activation, use_bias=True):
        self.shapes = shapes
        self.activation = activation
        self.use_bias= use_bias

        self.weights = np.random.uniform(-1, 1, shape=shapes)
        self.bias = np.random.uniform(-1, 1)

    def forward(self, input_data):
        self.output = np.dot(input_data, self.weights)
        if self.use_bias: 
            self.output += self.bias

        self.output = activation_functions(self.activation, self.output)

        return self.output