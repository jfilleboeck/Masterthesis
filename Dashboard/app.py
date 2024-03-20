import math

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import CubicSpline
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, f1_score

from data_preprocessing import load_and_preprocess_data
from model_adapter import ModelAdapter

# Load and split the data, determine if classification/regression
X_train, X_val, y_train, y_val, task = load_and_preprocess_data()


model = ModelAdapter(task)

model.fit(X_train, y_train)

# Setup
app = Flask(__name__)


def load_data():
    global shape_functions_dict, feature_history, feature_current_state
    shape_functions_dict = model.get_shape_functions_as_dict()
    feature_history = {}
    feature_current_state = {}

    for feature in shape_functions_dict:
        name = feature['name']
        y_value = feature['y']

        feature_history[name] = [y_value]
        feature_current_state[name] = y_value


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

    feature_name, x_data, y_data, is_numeric_feature, hist_data, bin_edges = next(
        (feature['name'], feature['x'].astype(float).tolist(),
         feature['y'].astype(float).tolist(), feature['datatype'],
         feature['hist'].hist.tolist(), feature['hist'].bin_edges.tolist())
        for feature in shape_functions_dict
    )

    return render_template('index.html', feature_names=X_names, x_data=x_data,
                           y_data=y_data, displayed_feature=feature_name, is_numeric_feature=is_numeric_feature,
                           hist_data=hist_data, bin_edges=bin_edges)


@app.route('/feature_data', methods=['POST'])
def feature_data():
    data = request.json
    displayed_feature = data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    if feature_data:
        if feature_data['datatype'] == 'numerical':
            x_data = feature_data['x'].tolist()
            y_data = feature_current_state[displayed_feature].tolist()
            # Convert histogram data and bin_edges to list
            hist_data = feature_data['hist'].hist.tolist()
            bin_edges = feature_data['hist'].bin_edges.tolist()
            #display feature kann entfernet werden

            return jsonify({'is_numeric': True, 'x': x_data, 'y': y_data,
                            'displayed_feature': displayed_feature,
                            'hist_data': hist_data, 'bin_edges': bin_edges})
        else:
            x_data = feature_data['x']
            encoded_x_data = encode_categorical_data(x_data)
            y_data = feature_current_state[displayed_feature]
            y_data = [float(y) if isinstance(y, np.float32) else y for y in y_data]
            # Convert histogram data to list
            hist_data = to_float_list(feature_data['hist'][0])
            bin_edges = encode_categorical_data(feature_data['hist'][1])
            return jsonify({'is_numeric': False, 'original_values': x_data,
                            'x': encoded_x_data, 'y': y_data,
                            'hist_data': hist_data, 'bin_edges': bin_edges,
                            'displayed_feature': displayed_feature})

    else:
        return jsonify({'error': 'Feature not found'}), 404

def to_float_list(lst):
    float_list = []
    for item in lst:
        if torch.is_tensor(item):
            float_list.append(item.item())
        elif isinstance(item, list):
            for sub_item in item:
                if torch.is_tensor(sub_item):
                    float_list.append(sub_item.item())
    return float_list

@app.route('/setConstantValue', methods=['POST'])
def setConstantValue():
    data = request.json
    x1, x2, new_y, displayed_feature = data['x1'], data['x2'], float(data['new_y']), data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    if not feature_data:
        return jsonify({'error': 'Feature not found'}), 404

    y_data = feature_current_state[displayed_feature].copy()

    if feature_data['datatype'] == 'numerical':
        x_data = feature_data['x']
    else:
        x_data = encode_categorical_data(feature_data['x'])

    for i, x in enumerate(x_data):
        if x1 <= x <= x2:
            y_data[i] = new_y

    feature_history[displayed_feature].append(y_data)
    feature_current_state[displayed_feature] = y_data

    return jsonify({'y': [float(y) for y in y_data] if feature_data['datatype'] != 'numerical' else [float(y) for y in
                                                                                                     y_data.tolist()]})


@app.route('/setLinear', methods=['POST'])
def setLinear():
    data = request.json
    x1, x2, displayed_feature = data['x1'], data['x2'], data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)

    if not feature_data:
        return jsonify({'error': 'Feature not found'}), 404

    y_data = feature_current_state[displayed_feature].copy()

    if feature_data['datatype'] == 'numerical':
        x_data = feature_data['x']
    else:
        x_data = encode_categorical_data(feature_data['x'])

    # Find indices for x1 and x2
    index_x1 = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x1))
    index_x2 = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x2))

    # Ensure indices are in the correct order
    index_start, index_end = sorted([index_x1, index_x2])

    slope = (y_data[index_end] - y_data[index_start]) / (x_data[index_end] - x_data[index_start])

    # Update y values along the line
    for i in range(index_start, index_end + 1):
        y_data[i] = y_data[index_start] + slope * (x_data[i] - x_data[index_start])

    feature_history[displayed_feature].append(y_data)
    feature_current_state[displayed_feature] = y_data

    return jsonify({'y': y_data if feature_data['datatype'] != 'numerical' else y_data.tolist()})


def weighted_isotonic_regression(x_data, y_data, hist_counts, bin_edges, increasing=True):
    # Determine the bin index for each x_data point
    bin_indices = np.digitize(x_data, bin_edges, right=True).astype(int)
    bin_indices = np.clip(bin_indices, 1, len(hist_counts))

    weights = np.array([hist_counts[index - 1] for index in bin_indices])

    iso_reg = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    iso_reg.fit(x_data, y_data, sample_weight=weights)
    y_pred = iso_reg.predict(x_data)

    if increasing:
        if y_pred[0] < y_data[0]:
            y_pred[0] = y_data[0]
            # Ensure the rest of the predictions maintain monotonicity
            for i in range(1, len(y_pred)):
                y_pred[i] = max(y_pred[i], y_pred[i-1])
    else:
        if y_pred[0] < y_data[0]:
            y_pred[0] = y_data[0]
            # Ensure the rest of the predictions maintain monotonicity
            for i in range(1, len(y_pred)):
                y_pred[i] = min(y_pred[i], y_pred[i-1])
    return y_pred





@app.route('/monotonic_increase', methods=['POST'])
def monotonic_increase():
    data = request.json
    displayed_feature = data['displayed_feature']
    x1, x2 = data['x1'], data['x2']
    y_data_full = feature_current_state[displayed_feature].copy()

    selected_item = next(item for item in shape_functions_dict if item['name'] == displayed_feature)
    # Numpy arrays are required for the IsotonicRegression package
    x_data = np.array(selected_item['x'])
    hist_data = np.array(selected_item['hist'].hist)
    bin_edges = np.array(selected_item['hist'].bin_edges)

    indices = np.where((x_data >= x1) & (x_data <= x2))[0]
    y_pred_subset = weighted_isotonic_regression(
        x_data[indices], y_data_full[indices], hist_data, bin_edges, increasing=True)

    y_data_full[indices] = y_pred_subset

    # Update feature history and current state
    feature_history[displayed_feature].append(feature_current_state[displayed_feature])
    feature_current_state[displayed_feature] = y_data_full

    return jsonify({'y': y_data_full.tolist()})


@app.route('/monotonic_decrease', methods=['POST'])
def monotonic_decrease():
    data = request.json
    displayed_feature = data['displayed_feature']
    x1, x2 = data['x1'], data['x2']
    y_data_full = feature_current_state[displayed_feature].copy()

    selected_item = next(item for item in shape_functions_dict if item['name'] == displayed_feature)
    # Numpy arrays are required for the IsotonicRegression package
    x_data = np.array(selected_item['x'])
    hist_data = np.array(selected_item['hist'].hist)
    bin_edges = np.array(selected_item['hist'].bin_edges)

    indices = np.where((x_data >= x1) & (x_data <= x2))[0]
    y_pred_subset = weighted_isotonic_regression(
        x_data[indices], y_data_full[indices], hist_data, bin_edges, increasing=False)

    y_data_full[indices] = y_pred_subset

    feature_history[displayed_feature].append(feature_current_state[displayed_feature])
    feature_current_state[displayed_feature] = y_data_full

    return jsonify({'y': y_data_full.tolist()})

@app.route('/setSmooth', methods=['POST'])
def setSmooth():
    data = request.json
    x1, x2, displayed_feature = data['x1'], data['x2'], data['displayed_feature']
    window_size = 5
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    data = request.json
    x1, x2, displayed_feature = data['x1'], data['x2'], data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    if not feature_data:
        return jsonify({'error': 'Feature not found'}), 404


    y_data = feature_current_state[displayed_feature].copy()

    if feature_data['datatype'] == 'numerical':
        x_data = feature_data['x']
    else:
        x_data = encode_categorical_data(feature_data['x'])

    # Find indices for x1 and x2
    index_start = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x1))
    index_end = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x2))
    #index_start, index_end = sorted([index_x1, index_x2])

    # Simple Moving Average
    smoothed_y = y_data.copy()
    for i in range(index_start, index_end + 1):
        window_indices = range(max(i - window_size // 2, 0), min(i + window_size // 2 + 1, len(y_data)))
        smoothed_y[i] = sum(y_data[j] for j in window_indices) / len(window_indices)

    feature_history[displayed_feature].append(smoothed_y)
    feature_current_state[displayed_feature] = smoothed_y

    return jsonify({'y': smoothed_y if feature_data['datatype'] != 'numerical' else smoothed_y.tolist()})


@app.route('/cubic_spline_interpolate', methods=['POST'])
def cubic_spline_interpolate():
    data = request.json
    selectedFeatures = data['selectedFeatures']
    displayed_feature = data['displayed_feature']
    #feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)

    updated_data = {}
    for feature in shape_functions_dict:
        name = feature['name']
        x_values = feature['x']
        if name in selectedFeatures:
            y_values = np.array(feature_current_state[name])
        else:
            y_values = feature['y']
    #feature_data_dict = next(item for item in shape_functions_dict if item['name'] == displayed_feature)
        if feature['datatype'] == 'numerical':
            updated_data[name] = {'x': x_values, 'y': y_values.tolist(), 'datatype': 'numerical'}
        else:
            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'categorical'}
    #x_data = feature_data_dict['x'].tolist()
    #y_data = feature_current_state[displayed_feature].tolist()
    model.adapt(selectedFeatures, updated_data, "spline_interpolation")
    #shape_functions_dict = model.model.get_shape_functions_as_dict()
    #load_data()
    #eplace_model(adapter)
    displayed_feature_data = next((item for item in model.get_shape_functions_as_dict() if item['name'] == displayed_feature), None)

    x_data = displayed_feature_data['x']
    y_data = displayed_feature_data['y']
    x_data_to_return = x_data.tolist() if not isinstance(x_data, list) else x_data
    y_data_to_return = y_data.tolist() if not isinstance(y_data, list) else y_data
    x_data_to_return = [float(x) for x in x_data_to_return]
    y_data_to_return = [float(y) for y in y_data_to_return]


    return jsonify({'x': x_data_to_return, 'y': y_data_to_return})

# def replace_model(new_model):
#     model = new_model
#     global shape_functions_dict
#     shape_functions_dict = model.get_shape_functions_as_dict()


@app.route('/retrain_feature', methods=['POST'])
def retrain_feature():
    data = request.json
    selectedFeatures = data['selectedFeatures']
    displayed_feature = data['displayed_feature']
    elmScale = data['elmScale']
    elmAlpha = data['elmAlpha']
    nrSyntheticDataPoints = data['nrSyntheticDataPoints']


    print(selectedFeatures)

    updated_data = {}
    for feature in shape_functions_dict:
        name = feature['name']
        x_values = feature['x']
        if name in selectedFeatures:
            y_values = np.array(feature_current_state[name])
        else:
            y_values = feature['y']
        # feature_data_dict = next(item for item in shape_functions_dict if item['name'] == displayed_feature)
        if feature['datatype'] == 'numerical':
            updated_data[name] = {'x': x_values, 'y': y_values.tolist(), 'datatype': 'numerical'}
        else:
            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'categorical'}
    # x_data = feature_data_dict['x'].tolist()
    # y_data = feature_current_state[displayed_feature].tolist()
    model.adapt(selectedFeatures, updated_data, "feature_retraining", (elmScale, elmAlpha, nrSyntheticDataPoints))
    # shape_functions_dict = model.model.get_shape_functions_as_dict()
    # load_data()
    # replace_model(adapter)
    displayed_feature_data = next(
        (item for item in model.get_shape_functions_as_dict() if item['name'] == displayed_feature), None)

    x_data = displayed_feature_data['x']
    y_data = displayed_feature_data['y']
    x_data_to_return = x_data.tolist() if not isinstance(x_data, list) else x_data
    y_data_to_return = y_data.tolist() if not isinstance(y_data, list) else y_data
    x_data_to_return = [float(x) for x in x_data_to_return]
    y_data_to_return = [float(y) for y in y_data_to_return]


    return jsonify({'x': x_data_to_return, 'y': y_data_to_return})




@app.route('/predict_and_get_metrics', methods=['GET'])
def predict_and_get_metrics():
    # Example model predictions and task definition
    # These should be replaced with your actual model predictions and task determination logic
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    if task == "regression":
        train_score = mean_squared_error(y_train, y_train_pred)
        val_score = mean_squared_error(y_val, y_val_pred)
    else:
        train_score = f1_score(y_train, y_train_pred, average='weighted')
        val_score = f1_score(y_val, y_val_pred, average='weighted')

    # Ensure the key names match what your frontend expects
    return jsonify({'train_score': train_score, 'val_score': val_score, 'task': task})



@app.route('/get_original_data', methods=['POST'])
def get_original_data():
    global feature_history, feature_current_state, feature_spline_state

    data = request.json
    displayed_feature = data['displayed_feature']

    # Obtain the original data for the selected feature
    model = ModelAdapter(task)
    model.fit(X_train, y_train)
    load_data()
    original_data = next(item for item in model.get_shape_functions_as_dict() if item['name'] == displayed_feature)
    original_y = original_data['y']

    # Reset the current state and history for the selected feature
    feature_current_state[displayed_feature] = original_y
    feature_history[displayed_feature] = [original_y.copy()]  # Reset history with the original state


    # Prepare data for response
    x_data = original_data['x']
    y_data = original_y
    x_data_to_return = x_data.tolist() if not isinstance(x_data, list) else x_data
    y_data_to_return = y_data.tolist() if not isinstance(y_data, list) else y_data
    x_data_to_return = [float(x) for x in x_data_to_return]
    y_data_to_return = [float(y) for y in y_data_to_return]

    return jsonify({'x': x_data_to_return, 'y': y_data_to_return})


@app.route('/undo_last_change', methods=['POST'])
def undo_last_change():
    data = request.json
    displayed_feature = data['displayed_feature']

    if displayed_feature in feature_history and len(feature_history[displayed_feature]) > 1:
        feature_current_state[displayed_feature] = feature_history[
            displayed_feature].pop()  # Revert to the previous state and remove the last change
        y_data = feature_current_state[displayed_feature]  # Update y_data to the reverted state
        return jsonify({'y': y_data.tolist()})
    else:
        return jsonify({'error': 'No more changes to undo for feature ' + displayed_feature}), 400


@app.route('/load_data_grid_instances', methods=['POST'])
def load_data_grid_instances():
    X_val_preprocessed = model.model._preprocess_feature_matrix(X_val, fit_dummies=True)
    X_val_preprocessed_df = pd.DataFrame(X_val_preprocessed.numpy())

    # Round numerical values to three decimal places
    X_val_preprocessed_df = X_val_preprocessed_df.round(3)

    y_val_reset = y_val.reset_index(drop=True)

    # Concatenate along the columns to get a single DataFrame
    combined_data = pd.concat([X_val_preprocessed_df, y_val_reset], axis=1)

    # Ensure that the target variable (or any other numerical columns in y_val_reset) is rounded as well
    combined_data = combined_data.round(3)

    combined_data.insert(0, 'ID', combined_data.index)

    # Convert DataFrame to dictionary with rounded numerical values
    rows = combined_data.to_dict(orient='records')
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 3)  # Ensure rounding persists in the dictionary

    for i, feature_name in enumerate(model.feature_names):
        for row in rows:
            if i in row:
                row[feature_name] = row.pop(i)

    return jsonify(rows)

@app.route('/instance_explanation', methods=['POST'])
def instance_explanation():
    data = request.json['data']
    selectedRowId = request.json['selectedRowId']
    # mit Auswahl Button die prediction für diese Zeile durchführen (wenn ausgewählt, dann wird auch predicted)
    # per Rechtsklick kann man den Auswahl Button triggern
    # {
    #     "ID": 0,
    #     "age": 0.953,
    #     "bmi": -0.13,
    #     "bp": -0.336,
    #     "s1": 2.628,
    #     "s2": 2.632,
    #     "s3": 0.403,
    #     "s4": 0.721,
    #     "s5": 0.682,
    #     "s6": -0.11,
    #     "sex_w": 0,
    #     "target": 0.867
    # }
    intercept = model.model.init_classifier.intercept_
    # als erstes ID und target cutten

    # für jedes dieser features den x wert nehmen und i mittels feature names bestimmen

    if task == "classification":
        contribution = torch.tensor(model.model.init_classifier.coef_[0, i] * x.numpy(), dtype=torch.float64)
        # y_hat = torch.tensor(y_hat, dtype=torch.float64)
    else:
        # + self.init_classifier.intercept_
        contribution = model.model.init_classifier.coef_[i] * x.numpy()

    # zurückschicken ans frontend: dictionary aus

def calculate_distance(row1, row2):
    distance = 0.0
    keys = list(row1.keys())
    # Exclude the first ('ID') and last columns from the keys to be considered in the calculation
    for key in keys[1:-1]:
        if key in row2:
            distance += (row1[key] - row2[key]) ** 2
    return math.sqrt(distance)

@app.route('/path/to/order_by_nearest', methods=['POST'])
def order_by_nearest():
    data = request.json['data']
    selectedRowId = request.json['selectedRowId']
    selectedRow = next(item for item in data if item['ID'] == selectedRowId)

    # Calculate distances to the selected row and sort
    for item in data:
        item['distance'] = calculate_distance(item, selectedRow)
    orderedData = sorted(data, key=lambda x: x['distance'])

    return jsonify(orderedData)

if __name__ == '__main__':
    app.run(debug=True)