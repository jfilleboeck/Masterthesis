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
            print(jsonify({'is_numeric': True, 'x': x_data, 'y': y_data,
                            'displayed_feature': displayed_feature,
                            'hist_data': hist_data, 'bin_edges': bin_edges}))
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

    return jsonify({'y': y_data if feature_data['datatype'] != 'numerical' else y_data.tolist()})


@app.route('/setLinear', methods=['POST'])
def setLinear():
    data = request.json
    x1, x2, displayed_feature = data['x1'], data['x2'], data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    print(feature_data)
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

    print(index_start, index_end, slope)
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
    return jsonify({'x': x_data.tolist(), 'y': y_data.tolist()})


# def replace_model(new_model):
#     model = new_model
#     global shape_functions_dict
#     shape_functions_dict = model.get_shape_functions_as_dict()


@app.route('/retrain_feature', methods=['POST'])
def retrain_feature():
    data = request.json
    selectedFeatures = data['selectedFeatures']
    displayed_feature = data['displayed_feature']

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
    model.adapt(selectedFeatures, updated_data, "feature_retraining")
    # shape_functions_dict = model.model.get_shape_functions_as_dict()
    # load_data()
    # replace_model(adapter)
    displayed_feature_data = next(
        (item for item in model.get_shape_functions_as_dict() if item['name'] == displayed_feature), None)

    x_data = displayed_feature_data['x']
    y_data = displayed_feature_data['y']
    return jsonify({'x': x_data.tolist(), 'y': y_data.tolist()})





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

    # Resetting feature_spline_state if necessary
    feature_spline_state[displayed_feature] = original_y

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


@app.route('/order_by_nearest', methods=['POST'])
def order_by_nearest():
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    # Step 1: Identify categorical and numeric columns
    categorical_features = X_val.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X_val.select_dtypes(exclude=['object']).columns.tolist()

    # Step 2: One-Hot Encode categorical columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(X_val[categorical_features])
    categorical_encoded_df = pd.DataFrame(categorical_encoded,
                                          columns=encoder.get_feature_names_out(categorical_features),
                                          index=X_val.index)  # Retain original index

    # Combine encoded categorical data with numeric data for nearest neighbor analysis
    X_val_numeric = X_val[numeric_features]
    X_val_preprocessed = pd.concat([X_val_numeric, categorical_encoded_df], axis=1)

    # Step 3: Fit NearestNeighbors and calculate neighbors
    nbrs = NearestNeighbors(n_neighbors=len(X_val_preprocessed)).fit(X_val_preprocessed)
    distances, indices = nbrs.kneighbors(X_val_preprocessed)

    # Reorder X_val and y_val based on the sorted indices of the nearest neighbors
    sorted_indices = indices[:, 1]  # Assuming you want to sort based on the closest neighbor
    # Use original index to ensure alignment
    original_indices = X_val_preprocessed.index[sorted_indices]
    sorted_data_original = X_val.loc[original_indices]

    # Align y_val with the sorted X_val
    sorted_y_val = y_val.loc[original_indices]

    # Convert the sorted data to JSON, including y_val
    sorted_data_original['target'] = sorted_y_val.values  # Direct assignment using values to avoid index misalignment
    sorted_data_json = sorted_data_original.to_dict(orient='records')

    return jsonify(sorted_data_json)

if __name__ == '__main__':
    app.run(debug=True)