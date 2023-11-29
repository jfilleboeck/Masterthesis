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


if __name__ == "__main__":
    print("Running main script")

# Load and split the data
X_train, X_test, y_train, y_test = load_and_preprocess_data()
model = ModelAdapter()

#model = IGANN(task='regression')
model.fit(X_train, y_train)



# Initial data load
shape_functions_dict = model.get_shape_functions_as_dict()
feature_history = {feature['name']: [feature['y']] for feature in shape_functions_dict}
feature_current_state = {feature['name']: feature['y'] for feature in shape_functions_dict}
feature_spline_state = {feature['name']: feature['y'] for feature in shape_functions_dict}



# List of features to update
features_to_update = ['age']

# Loop over each feature in the list and update its y values in feature_spline_state
for feature_name in features_to_update:
    if feature_name in feature_spline_state:
        # Hier müssen die Werte aus dem Frontend eingetragen werden
        feature_spline_state[feature_name] = np.full_like(feature_spline_state[feature_name], 0.05)


updated_data = [feature_spline_state[feature] for feature in features_to_update if feature in feature_spline_state]
#cs = CubicSpline(feature_dict['x'], feature_spline_state[feature])
#updated_data.append(cs)
model = model.adapt(selected_features=features_to_update, updated_data=updated_data, X_train=X_train, y_train=y_train, method="spline")
model.predict(X_test)
#cs = CubicSpline(x, y)
print(type(model))