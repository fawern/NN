import pandas as pd 
import numpy as np 
np.random.seed(0)

from sklearn.model_selection import train_test_split

from Nn import Layers
from Nn import NLayer
from Nn import NInput

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

data_generator = DataGenerator(5)
# data_generator.add_feature([1, 2, 3, 4, 5, 6, 7])

df = data_generator.generate_data()

X = df.drop(columns='gender').values
y = df['gender'].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = Layers()

# model.add(NInput(num_neurons=X_train.shape[1]))

## Input Layer 
model.add(NLayer(num_neurons=X_train.shape[1], use_bias=True)) # egitild

## Hidden Layers
model.add(NLayer(num_neurons=256, activation='softmax', use_bias=True)) 
model.add(NLayer(num_neurons=128, activation='softmax', use_bias=True)) 
model.add(NLayer(num_neurons=64, activation='sigmoid', use_bias=True)) 

## Output Layer
model.add(NLayer(num_neurons=1, activation='sigmoid', use_bias=False))

## Train model
model.train_model(x=X_train)

y_pred = model.output

print(y_pred)

def gender(output):
    return 'Female' if output > 0.5 else "Male"

print('\n')
for i, output in enumerate(y_pred):
    print(f'Prediction for input {i+1}: {gender(output)}, true value is {"Kadin" if y_pred[i] == 1 else "Erkek"}')