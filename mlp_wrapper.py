from Nn import Layers
from Nn import NInput
from Nn import NLayer

class LayersWrap(Layers):
    '''
    LayersWrap class is a wrapper class for the Layers class.
    '''
    def __init__(self, input_data, output_data, hl_activation, ol_activation):
        """
        Args:
            - hl_activation (str): The activation function of hidden layers.
            - ol_activation (str): The activatoin function of output layer
        """

        ## Initialize the layers class.
        Layers.__init__(self)

        self.input_data = input_data
        self.output_data = output_data
        self.hl_activation = hl_activation
        self.ol_activation = ol_activation

        ## Input Layer
        self.add(NInput(input_shape=self.input_data.shape[1]))

        ## Hidden Layers
        self.add(NLayer(num_neurons=64, activation=self.hl_activation, use_bias=True))
        self.add(NLayer(num_neurons=32, activation=self.hl_activation, use_bias=True))
        self.add(NLayer(num_neurons=8, activation=self.hl_activation, use_bias=True)) 

        ## Output Layer
        self.add(NLayer(num_neurons=output_data.shape[1], activation=self.ol_activation, use_bias=False))

    def fit_model(self, learning_rate=0.1, iterations=1000000, batch_size=32):
        self.train_model(x=self.input_data, y=self.output_data, iterations=iterations, learning_rate=learning_rate, batch_size=batch_size)

    def get_accuracy(self):
        return self.output