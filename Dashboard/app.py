from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from igann import IGANN
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline

from data_preprocessing import load_and_preprocess_data
from model_adapter import ModelAdapter

app = Flask(__name__)

# Load and split the data
X_train, X_test, y_train, y_test = load_and_preprocess_data()
model = ModelAdapter()

#model = IGANN(task='regression')
model.fit(X_train, y_train)

# Setup
app = Flask(__name__)


# Initial data load
shape_functions_dict = model.get_shape_functions_as_dict()
feature_history = {feature['name']: [feature['y']] for feature in shape_functions_dict}
feature_current_state = {feature['name']: feature['y'] for feature in shape_functions_dict}
feature_spline_state = {feature['name']: feature['y'] for feature in shape_functions_dict}


@app.route('/')
def index():
    # Render with all features available to choose from
    X_names_list = X_train.columns.tolist()
    name_first_num, x_values_first_num, y_values_first_num = next(
        (feature['name'], feature['x'].astype(float).tolist(), feature['y'].astype(float).tolist())
        for feature in shape_functions_dict if feature['datatype'] == 'numerical'
    )

    return render_template('index.html', feature_names=X_names_list, x_data=x_values_first_num, y_data=y_values_first_num,
                           selected_feature=name_first_num)

@app.route('/feature_data', methods=['POST'])
def feature_data():
    data = request.json
    selected_feature = data['selected_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == selected_feature), None)
    if feature_data:
        x_data = feature_data['x'].tolist()
        y_data = feature_current_state[selected_feature].tolist()
        return jsonify({'x': x_data, 'y': y_data, 'selected_feature': selected_feature})
    else:
        return jsonify({'error': 'Feature not found'}), 404

@app.route('/setConstantValue', methods=['POST'])
def setConstantValue():
    data = request.json
    selected_feature = data['selected_feature']
    x1, x2, new_y = data['x1'], data['x2'], float(data['new_y'])

    y_data = feature_current_state[selected_feature]
    x_data = next(item for item in shape_functions_dict if item['name'] == selected_feature)['x'].tolist()

    history_entry = y_data.copy()
    for i, x in enumerate(x_data):
        if x1 <= x <= x2:
            y_data[i] = new_y

    # Append the history entry after making changes
    feature_history[selected_feature].append(history_entry)

    # Save the modified state back to the feature_current_state dictionary
    feature_current_state[selected_feature] = y_data

    return jsonify({'y': y_data.tolist()})


def find_nearest(array, value):
    """Find the index of the nearest value in the array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

@app.route('/monotonic_increase', methods=['POST'])
def monotonic_increase():
    data = request.json
    selected_feature = data['selected_feature']

    y_data = feature_current_state[selected_feature]
    x_data = next(item for item in shape_functions_dict if item['name'] == selected_feature)['x'].tolist()

    feature_history[selected_feature].append(y_data.copy())

    x1, x2 = data['x1'], data['x2']
    start_index = max(find_nearest(x_data, x1), 1)  # Ensure start_index is greater than 0
    end_index = find_nearest(x_data, x2)

    # Ensure increase is monotonic
    for i in range(start_index, end_index + 1):  # Include end_index in the loop
        y_data[i] = max(y_data[i], y_data[i-1])

    feature_current_state[selected_feature] = y_data

    return jsonify({'y': y_data.tolist()})


@app.route('/monotonic_decrease', methods=['POST'])
def monotonic_decrease():
    data = request.json
    selected_feature = data['selected_feature']  # Obtain the feature_name from the request

    y_data = feature_current_state[selected_feature]
    x_data = next(item for item in shape_functions_dict if item['name'] == selected_feature)['x'].tolist()

    # Record current state before making changes
    feature_history[selected_feature].append(y_data.copy())

    # Get start and end index for the selected x range
    x1, x2 = data['x1'], data['x2']
    start_index = find_nearest(x_data, x1)
    end_index = find_nearest(x_data, x2)

    # Ensure decrease is monotonic by only decreasing if the next point is not higher
    for i in range(end_index, start_index, -1):
        y_data[i - 1] = min(y_data[i - 1], y_data[i])

    # Make sure the starting point is not lower than the end point if they are out of order
    if y_data[start_index] < y_data[end_index]:
        y_data[start_index] = y_data[end_index]

    # Save the modified state
    feature_current_state[selected_feature] = y_data

    return jsonify({'y': y_data.tolist()})

@app.route('/cubic_spline_interpolate', methods=['POST'])
def cubic_spline_interpolate():
    data = request.json
    selected_feature = data['selected_feature']  # Obtain the feature name from the request

    # Access the original feature's x and y data from shape_functions_dict
    feature_data_dict = next(item for item in shape_functions_dict if item['name'] == selected_feature)
    x_data = feature_data_dict['x'].tolist()
    original_y_data = feature_data_dict['y'].tolist()

    # Perform cubic spline interpolation on the original y_data
    cs = CubicSpline(x_data, original_y_data)
    ynew = cs(x_data)

    # Update the feature_spline_state dictionary with the new spline data
    feature_spline_state[selected_feature] = ynew.tolist()

    # Return the interpolated data
    return jsonify({'x': x_data, 'y': ynew.tolist()})


@app.route('/predict_and_get_mse', methods=['GET'])
def predict_and_get_mse():
    # Use the global model to predict and calculate MSE
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Return the MSE values as JSON
    return jsonify({'mse_train': mse_train, 'mse_test': mse_test})


@app.route('/get_original_data', methods=['POST'])
def get_original_data():
    data = request.json
    selected_feature = data['selected_feature']
    #global x_data, y_data
    data_dict = next(item for item in model.get_shape_functions_as_dict() if item['name'] == selected_feature)
    x_data = data_dict['x'].tolist()
    y_data = data_dict['y'].tolist()
    return jsonify({'x': x_data, 'y': y_data})

@app.route('/undo_last_change', methods=['POST'])
def undo_last_change():
    data = request.json
    print(data)
    selected_feature = data['selected_feature']

    if selected_feature in feature_history and len(feature_history[selected_feature]) > 1:
        feature_history[selected_feature].pop()  # Remove the last change
        feature_current_state[selected_feature] = feature_history[selected_feature][-1]  # Revert to the previous state
        y_data = feature_current_state[selected_feature]  # Update y_data to the reverted state
        return jsonify({'y': y_data.tolist()})
    else:
        return jsonify({'error': 'No more changes to undo for feature ' + selected_feature}), 400


@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.json
    selected_feature = data['selected_feature']
    #TODO: Möglichkeit auswählen können
    method = "spline"
    list_of_features = [selected_feature]

    model.adapt(list_of_features, feature_spline_state, X_train, y_train, method="spline")


    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Return the MSE values as JSON
    return jsonify({'mse_train': mse_train, 'mse_test': mse_test})


if __name__ == '__main__':
    app.run(debug=True)