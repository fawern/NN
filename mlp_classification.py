import numpy as np 
import pandas as pd 

np.random.seed(0)

from mlp_wrapper import LayersWrap


df = pd.read_csv("./data/gender_classification_v7.csv")
np.random.shuffle(df.values)
df = df.iloc[:100]

# mapping gender to 0 and 1
df['gender'] = df['gender'].map({"Male" : 0 , "Female" : 1})

X = df.drop(columns='gender').values
y = df['gender'].values.reshape(-1, 1)

model = LayersWrap(input_data=X, output_data=y, hl_activation='sigmoid', ol_activation='sigmoid')

model.fit_model(loss_type='categorical', learning_rate=0.0001, iterations=100000, batch_size=1)

print(model.evaluate_trained_model())

model.show_loss_graph()