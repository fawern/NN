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

inputs = []
outputs = []

def generate_data(num_row):
    for i in range(num_row):
        height = np.random.uniform(0.5, 2.5)
        weight = np.random.randint(10, 200)

        # eye_colors = ['Blue', 'Green', 'Brown', 'Hazel', 'Gray', 'Amber', 'Black']
        eye_colors = [1, 2, 3, 4, 5, 6, 7]
        eye_color =  np.random.choice(eye_colors)

        # hair_colors = ['Black', 'Brown', 'Blonde', 'Red', 'Gray', 'White', 'Auburn']
        hair_colors = [1, 2, 3, 4, 5, 6, 7]
        hair_color = np.random.choice(hair_colors)

        output = np.random.randint(0, 2)


        inputs.append([height, weight, eye_color, hair_color])
        outputs.append(output)

    return inputs, outputs

inputs, outputs = generate_data(5)
inputs = np.array(inputs)
outputs = np.array(outputs)

# layer_1 = Dense(inputs.shape[1], 256, activation='softmax', use_bias=True)
# output_1 = layer_1.forward(inputs)

# layer_2 = Dense(256, 128, activation='softmax', use_bias=True)
# output_2 = layer_2.forward(output_1)

# layer_3 = Dense(128, 64, activation='softmax', use_bias=True)
# output_3 = layer_3.forward(output_2)

# layer_4 = Dense(64, 1, activation='sigmoid', use_bias=True) 
# output_4 = layer_4.forward(output_3)

def gender(output):
    return 'Kadin' if output > 0.5 else "Erkek"






model = Sequential()
# model.add(Dense(shapes=(inputs.shape[1], 256), activation_func='softmax', use_bias=True))
# model.add(Dense(shapes=(256, 128), activation_func='softmax', use_bias=True))
# model.add(Dense(shapes=(128, 64), activation_func='softmax', use_bias=True))
# model.add(Dense(shapes=(64, 1), activation_func='sigmoid', use_bias=True))

