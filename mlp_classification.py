import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(0)

from sklearn.model_selection import train_test_split

from Nn import Layers

from Nn import NInput
from Nn import NLayer

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1]])
y = np.array([[0], [1], [1], [0], [0], [0], [0], [1], [1], [0], [0], [0]])

df = pd.read_csv("./gender_classification_v7.csv")
print(df.tail())
# mapping gender to 0 and 1
df['gender'] = df['gender'].map({"Male" : 0 , "Female" : 1})

X = df.drop(columns='gender').values
y = df['gender'].values.reshape(-1, 1)

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
        self.add(NInput(num_neurons=self.input_data.shape[1]))

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

model = LayersWrap(
    input_data=X, output_data=y, 
    hl_activation='sigmoid', ol_activation='sigmoid', 
)

model.fit_model(learning_rate=0.01, iterations=100, batch_size=4)

print(model.evaluate_trained_model())

model.show_loss_graph()