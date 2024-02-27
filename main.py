import numpy as np 

from Nn import Layers
from Nn import NLayer

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

    return np.array(inputs), np.array(outputs)

X_train, y_train = generate_data(5)

model = Layers()

model.add(NLayer(shapes=(X_train.shape[1], 256), activation='softmax', use_bias=True))
model.add(NLayer(shapes=(256, 128), activation='softmax', use_bias=True))
model.add(NLayer(shapes=(128, 64), activation='softmax', use_bias=True))
model.add(NLayer(shapes=(64, 1), activation='sigmoid', use_bias=True))

model.train_model(x=X_train)

y_pred = model.predict_input()

def gender(output):
    return 'Kadin' if output > 0.5 else "Erkek"

print('\n')
for i, output in enumerate(y_pred):
    print(f'Prediction for input {i+1}: {gender(output)}, true value is {"Kadin" if outputs[i] == 1 else "Erkek"}')