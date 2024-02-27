import numpy as np 

def activation_functions(activation, x):

    if activation == 'softmax':
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(f"{activation_func} is not a valid activation function!!!")

class Layers:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def train_model(self, x, output_data=None):
        self.output = self.layers[0].forward(x)

        print("===============================================================")
        print(f"({1}/{len(self.layers)}) {1} Model trained")
        print("---------------------------------------------------------------")
        for i in range(1, len(self.layers)):
            self.output = self.layers[i].forward(self.output)
            print(f"({i+2}/{len(self.layers)}) {i+2} Model trained")
            print("---------------------------------------------------------------")
    
    def predict_input(self):
        return self.output

class NLayer:
    def __init__(self, shapes, activation, use_bias=True):
        self.shapes = shapes
        self.activation = activation
        self.use_bias= use_bias

        self.weights = np.random.uniform(-1, 1, size=self.shapes)
        self.bias = np.random.uniform(-1, 1)

    def forward(self, input_data):
        self.output = np.dot(input_data, self.weights)
        if self.use_bias: 
            self.output += self.bias

        self.output = activation_functions(self.activation, self.output)

        return self.output