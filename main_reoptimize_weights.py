import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from igann import IGANN
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline

from Dashboard.data_preprocessing import load_and_preprocess_data
from Dashboard.model_adapter import ModelAdapter

import warnings



if __name__ == "__main__":
    print("Running main script")

# Load and split the data
X_train, X_test, y_train, y_test = load_and_preprocess_data()
model = ModelAdapter()

#print(type(X_train))
#print(X_train['s1'])

#model = IGANN(task='regression')
model.fit(X_train, y_train)
model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])


# Initial data load
shape_functions_dict = model.get_shape_functions_as_dict()
feature_history = {feature['name']: [feature['y']] for feature in shape_functions_dict}
feature_current_state = {feature['name']: feature['y'] for feature in shape_functions_dict}
#feature_spline_state = {feature['name']: feature['y'] for feature in shape_functions_dict}



# List of features to update
features_to_update = ['s1']

# Get index of s1
#s1_index = model.feature_names.index('s1')

# parabola values
dict_data = next(item for item in model.get_shape_functions_as_dict() if item['name'] == 's1')
x = dict_data['x']
#x = model.unique[s1_index]
a, b, c = 0.05, 0, 0
y = a * x**2 + b * x + c

# plt.figure(figsize=(8, 6))
# plt.plot(x, y, label='y = 0.05x^2')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of the Parabola y = 0.05x^2')
# plt.legend()
# plt.grid(True)
# plt.show()


updated_data = {'s1': y}



y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(mse_train)
print(mse_test)

model = model.adapt(selected_features=features_to_update, updated_data=updated_data, X_train=X_train, y_train=y_train, method="reoptimize_weights")
model.predict(X_test)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(mse_train)
print(mse_test)

model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])