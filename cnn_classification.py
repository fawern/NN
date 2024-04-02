import numpy as np 
import pandas as pd 

from tensorflow.keras.datasets import mnist

from Nn import NInput  
from Nn import ConvNLayer
from Nn import NLayer


(X_train, y_train), (X_test, y_test) = mnist.load_data()