import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
X_train, X_test, y_train, y_test, task = load_and_preprocess_data()

model = IGANN(task)


#model = IGANN(task='regression')
model.fit(X_train, y_train)
#model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])


print(model.get_shape_functions_as_dict())

model.plot_single(plot_by_list=['Pclass', 'Sex', 'Embarked'])