import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(0)

from mlp_wrapper import LayersWrap

df = pd.read_csv("./data/train.csv")
df = df.dropna()

X = df['x'].values.reshape(-1, 1)
y = df['y'].values.reshape(-1, 1)

plt.scatter(X, y)
plt.show()

model = LayersWrap(input_data=X, output_data=y, hl_activation='softmax', ol_activation='linear', )

model.fit_model(loss_type='mae', learning_rate=0.01, iterations=2000, batch_size=1)

print(model.evaluate_trained_model())

model.show_loss_graph()