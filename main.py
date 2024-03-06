import pandas as pd 
import numpy as np 
# np.random.seed(0)

from sklearn.model_selection import train_test_split

from Nn import Layers

from Nn import NInput
from Nn import NLayer

inputs = []
outputs = []

class DataGenerator:
    def __init__(self, num_row):
        self.num_row = num_row
        self.features = ['height', 'weight', 'eye_color', 'hair_color', 'gender']
        self.df = pd.DataFrame(columns=self.features)

    def add_feature(self, feature):
        self.features.append(feature)

    def generate_data(self):
        for i in range(self.num_row):

            height = np.random.uniform(0.5, 2.5)
            weight = np.random.randint(10, 200)

            # eye_colors = ['Blue', 'Green', 'Brown', 'Hazel', 'Gray', 'Amber', 'Black']
            eye_colors = [1, 2, 3, 4, 5, 6, 7]
            eye_color =  np.random.choice(eye_colors)

            # hair_colors = ['Black', 'Brown', 'Blonde', 'Red', 'Gray', 'White', 'Auburn']
            hair_colors = [1, 2, 3, 4, 5, 6, 7]
            hair_color = np.random.choice(hair_colors)  

            output = np.random.randint(0, 2)

            self.df.loc[len(self.df)] = [height, weight, eye_color, hair_color, output]

        return self.df

    @staticmethod
    def train_test_split(X, y, train_rate):
        num_train = int(len(X) * train_rate)
        indices = np.random.permutation(len(X))
        train_indices, test_indices = indices[:num_train], indices[num_train:]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test

data_generator = DataGenerator(1000)
# data_generator.add_feature([1, 2, 3, 4, 5, 6, 7])
df = data_generator.generate_data()

X = df.drop(columns='gender').values
y = df['gender'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

class LayersWrap(Layers):
    '''
    LayersWrap class is a wrapper class for the Layers class.
    '''
    def __init__(self, hl_activation, ol_activation):

        """
        Args:
            - hl_activation (str): The activation function of hidden layers.
            - ol_activation (str): The activatoin function of output layer
        """

        ## Initialize the layers class.
        Layers.__init__(self)

        self.hl_activation = hl_activation
        self.ol_activation = ol_activation

        ## Input Layer
        self.add(NInput(num_neurons=X_train.shape[1]))

        ## Hidden Layers
        self.add(NLayer(num_neurons=256, activation=self.hl_activation, use_bias=True)) 
        self.add(NLayer(num_neurons=128, activation=self.hl_activation, use_bias=True)) 
        self.add(NLayer(num_neurons=64, activation=self.hl_activation, use_bias=True)) 

        ## Output Layer
        self.add(NLayer(num_neurons=1, activation=self.ol_activation, use_bias=False))

        ## Train model
        self.train_model(x=X_train, y=y_train, iterations=6)

    def get_accuracy(self):
        return self.output

model = LayersWrap(hl_activation='softmax', ol_activation='sigmoid')

y_output = model.get_accuracy()

print(f'Accuracy of model : {model.evaluate_trained_model()}')