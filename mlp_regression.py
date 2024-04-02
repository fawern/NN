import numpy as np 
import pandas as pd

np.random.seed(0)

from mlp_wrapper import LayersWrap

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).reshape(-1, 1)

model = LayersWrap(input_data=X, output_data=y, hl_activation='tanh', ol_activation='linear', )

model.fit_model(loss_type='mse', learning_rate=0.0001, iterations=1000, batch_size=1)

print(model.evaluate_trained_model())

model.show_loss_graph()