import numpy as np 
import pandas as pd 

from tensorflow.keras.datasets import mnist

from Nn import Layers
from Nn import NInput  
from Nn import ConvNLayer
from Nn import FlattenLayer
from Nn import NLayer


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


model = Layers()

model.add(NInput(input_shape=(28, 28)))
model.add(ConvNLayer(num_neurons=128, activation='relu'))
model.add(ConvNLayer(num_neurons=64, activation='relu'))

model.add(FlattenLayer())

model.add(NLayer(num_neurons=10, activation='softmax'))
