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

# Load and split the data, determine if classification/regression
X_train, X_val, y_train, y_val, task = load_and_preprocess_data(dataset='diabetes')

model = ModelAdapter(task)

model.fit(X_train, y_train)

# Setup
app = Flask(__name__)


def load_data():
    global shape_functions_dict, feature_history, feature_current_state, feature_spline_state
    shape_functions_dict = model.get_shape_functions_as_dict()
    feature_history = {}
    feature_current_state = {}
    feature_spline_state = {}

    for feature in shape_functions_dict:
        name = feature['name']
        y_value = feature['y']

        feature_history[name] = [y_value]
        feature_current_state[name] = y_value
        feature_spline_state[name] = y_value


def encode_categorical_data(categories):
    # Encode each category based on its index
    encoded_values = [index for index, category in enumerate(categories)]

    return encoded_values



# Initial data load from original model
load_data()


@app.route('/')
def index():
    # Render with all features available to choose from
    X_names = X_train.columns.tolist()

    feature_name, x_data, y_data, is_numeric_feature = next(
        (feature['name'], feature['x'].astype(float).tolist(), feature['y'].astype(float).tolist(), feature['datatype'])
        for feature in shape_functions_dict
    )

    return render_template('index.html', feature_names=X_names, x_data=x_data,
                           y_data=y_data, selected_feature=feature_name, is_numeric_feature=is_numeric_feature)


@app.route('/feature_data', methods=['POST'])
def feature_data():
    data = request.json
    selected_feature = data['selected_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == selected_feature), None)
    if feature_data:
        if feature_data['datatype'] == 'numerical':
            x_data = feature_data['x'].tolist()
            y_data = feature_current_state[selected_feature].tolist()
            return jsonify({'is_numeric': True, 'x': x_data, 'y': y_data,
                            'selected_feature': selected_feature})
        else:
            x_data = feature_data['x']
            encoded_x_data = encode_categorical_data(x_data)
            y_data = feature_current_state[selected_feature]
            y_data = [float(y) if isinstance(y, np.float32) else y for y in y_data]
            return jsonify({'is_numeric': False, 'original_values': x_data,
                            'x': encoded_x_data, 'y': y_data, 'selected_feature': selected_feature})
    else:
        return jsonify({'error': 'Feature not found'}), 404


@app.route('/setConstantValue', methods=['POST'])
def setConstantValue():
    data = request.json
    x1, x2, new_y, selected_feature = data['x1'], data['x2'], float(data['new_y']), data['selected_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == selected_feature), None)
    # y_data = feature_current_state[selected_feature]
    y_data = feature_current_state[selected_feature].copy()
    print(type(feature_data['x']))
    # history_entry = y_data.copy()
    if feature_data['datatype'] == 'numerical':
        x_data = feature_data['x'].tolist()
        for i, x in enumerate(x_data):
            if x1 <= x <= x2:
                y_data[i] = new_y
        feature_history[selected_feature].append(feature_current_state[selected_feature])

        feature_current_state[selected_feature] = y_data

        return jsonify({'y': y_data.tolist()})

    else:
        x_data = feature_data['x']
        encoded_x_data = encode_categorical_data(x_data)
        for i, x in enumerate(encoded_x_data):
            if x1 <= x <= x2:
                y_data[i] = new_y
        feature_history[selected_feature].append(feature_current_state[selected_feature])

        feature_current_state[selected_feature] = y_data

        return jsonify({'y': y_data})




def find_nearest(array, value):
    """Find the index of the nearest value in the array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


@app.route('/monotonic_increase', methods=['POST'])
def monotonic_increase():
    data = request.json
    selected_feature = data['selected_feature']

    y_data = feature_current_state[selected_feature].copy()
    x_data = next(item for item in shape_functions_dict if item['name'] == selected_feature)['x'].tolist()

    # feature_history[selected_feature].append(y_data.copy())
    feature_history[selected_feature].append(feature_current_state[selected_feature])

    x1, x2 = data['x1'], data['x2']
    start_index = max(find_nearest(x_data, x1), 1)  # Ensure start_index is greater than 0
    end_index = find_nearest(x_data, x2)

    for i in range(start_index + 1, end_index):
        y_data[i] = max(y_data[i], y_data[i - 1])

    feature_current_state[selected_feature] = y_data

    return jsonify({'y': y_data.tolist()})


@app.route('/monotonic_decrease', methods=['POST'])
def monotonic_decrease():
    data = request.json
    selected_feature = data['selected_feature']

    y_data = feature_current_state[selected_feature].copy()
    x_data = next(item for item in shape_functions_dict if item['name'] == selected_feature)['x'].tolist()

    feature_history[selected_feature].append(feature_current_state[selected_feature])

    x1, x2 = data['x1'], data['x2']
    start_index = find_nearest(x_data, x1)
    end_index = find_nearest(x_data, x2)

    for i in range(end_index, start_index, -1):
        y_data[i - 1] = min(y_data[i - 1], y_data[i])

    if y_data[start_index] < y_data[end_index]:
        y_data[start_index] = y_data[end_index]

    # Save the modified state
    feature_current_state[selected_feature] = y_data

    return jsonify({'y': y_data.tolist()})


@app.route('/cubic_spline_interpolate', methods=['POST'])
def cubic_spline_interpolate():
    data = request.json
    selected_feature = data['selected_feature']

    feature_data_dict = next(item for item in shape_functions_dict if item['name'] == selected_feature)
    x_data = feature_data_dict['x'].tolist()
    y_data = feature_current_state[selected_feature].tolist()

    cs = CubicSpline(x_data, y_data)
    ynew = cs(x_data)

    feature_spline_state[selected_feature] = ynew.tolist()

    return jsonify({'x': x_data, 'y': ynew.tolist()})


@app.route('/predict_and_get_mse', methods=['GET'])
def predict_and_get_mse():
    # Use the global model to predict and calculate MSE
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)

    # Return the MSE values as JSON
    return jsonify({'mse_train': mse_train, 'mse_val': mse_val})


@app.route('/get_original_data', methods=['POST'])
def get_original_data():
    global feature_history, feature_current_state, feature_spline_state

    data = request.json
    selected_feature = data['selected_feature']

    # Obtain the original data for the selected feature
    model = ModelAdapter()
    model.fit(X_train, y_train)
    original_data = next(item for item in model.get_shape_functions_as_dict() if item['name'] == selected_feature)
    original_y = original_data['y']

    # Reset the current state and history for the selected feature
    feature_current_state[selected_feature] = original_y
    feature_history[selected_feature] = [original_y.copy()]  # Reset history with the original state

    # Resetting feature_spline_state if necessary
    feature_spline_state[selected_feature] = original_y

    # Prepare data for response
    x_data = original_data['x'].tolist()
    y_data = original_y.tolist()

    return jsonify({'x': x_data, 'y': y_data})


@app.route('/undo_last_change', methods=['POST'])
def undo_last_change():
    data = request.json
    selected_feature = data['selected_feature']

    if selected_feature in feature_history and len(feature_history[selected_feature]) > 1:
        feature_current_state[selected_feature] = feature_history[
            selected_feature].pop()  # Revert to the previous state and remove the last change
        y_data = feature_current_state[selected_feature]  # Update y_data to the reverted state
        return jsonify({'y': y_data.tolist()})
    else:
        return jsonify({'error': 'No more changes to undo for feature ' + selected_feature}), 400


@app.route('/update_weights', methods=['POST'])
def update_weights():
    data = request.json
    selected_feature = data['selected_feature']
    feature_data_dict = next(item for item in model.get_shape_functions_as_dict() if item['name'] == selected_feature)
    print(model.feature_names)
    print(selected_feature)
    y_data = feature_current_state[selected_feature]
    print(y_data)
    model.adapt(selected_feature, y_data, "reoptimize_weights", X_train, y_train)
    # TODO: Archivierung
    # TODO: Momentan werden alle Features zurückgesetzt

    feature_data_dict = next(item for item in model.get_shape_functions_as_dict() if item['name'] == selected_feature)
    print(feature_data_dict)
    feature_current_state[selected_feature] = feature_data_dict['y']

    return jsonify({'y': feature_current_state[selected_feature].tolist()})


@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.json
    selected_feature = data['selected_feature']
    # TODO: Möglichkeit auswählen können
    method = "spline"
    list_of_features = [selected_feature]

    model.adapt(list_of_features, feature_spline_state, X_train, y_train, method="spline")

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)

    return jsonify({'mse_train': mse_train, 'mse_val': mse_val})

@app.route('/load_data_grid_instances', methods=['POST'])
def load_data_grid_instances():
    data = request.json
    if data and data.get('type_of_data') == 'initial':
        combined_data = pd.concat([X_val, y_val], axis=1)
        combined_data = combined_data.round(3)
        rows = combined_data.to_dict(orient='records')

        return jsonify(rows)

    return jsonify({'error': 'Invalid request'}), 400



if __name__ == '__main__':
    app.run(debug=True)
