from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from igann import IGANN
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline


X, y = load_diabetes(return_X_y=True, as_frame=True)
scaler = StandardScaler()
X_names = X.columns

# Setup
app = Flask(__name__)
scaler = StandardScaler()

# Load data
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_names = X.columns
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X_names)
X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')
y = (y - y.mean()) / y.std()

# Split dataset and create model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = IGANN(task='regression')
model.fit(X_train, y_train)

# Initial data load
shape_functions_dict = model.get_shape_functions_as_dict()
feature_history = {feature['name']: [feature['y']] for feature in shape_functions_dict}
feature_current_state = {feature['name']: feature['y'] for feature in shape_functions_dict}




@app.route('/')
def index():
    # Render with all features available to choose from
    X_names_list = X_names.tolist()

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
    print(data)
    if feature_data:
        x_data = feature_data['x'].tolist()
        print(x_data)
        y_data = feature_current_state[selected_feature].tolist()
        return jsonify({'x': x_data, 'y': y_data, 'selected_feature': selected_feature})
    else:
        return jsonify({'error': 'Feature not found'}), 404

@app.route('/setConstantValue', methods=['POST'])
def setConstantValue():
    data = request.json
    print(data)
    selected_feature = data['selected_feature']
    x1, x2, new_y = data['x1'], data['x2'], float(data['new_y'])
    print(data)

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
    history.append(y_data.copy())
    data = request.json
    x1, x2 = data['x1'], data['x2']
    start_index = find_nearest(x_data, x1)
    end_index = find_nearest(x_data, x2)

    interpolated_values = np.linspace(y_data[start_index], y_data[end_index], end_index - start_index + 1)

    for i, y_val in zip(range(start_index, end_index + 1), interpolated_values):
        y_data[i] = y_val

    return jsonify({'y': y_data})

@app.route('/monotonic_decrease', methods=['POST'])
def monotonic_decrease():
    history.append(y_data.copy())
    data = request.json
    x1, x2 = data['x1'], data['x2']
    start_index = find_nearest(x_data, x1)
    end_index = find_nearest(x_data, x2)

    interpolated_values = np.linspace(y_data[start_index], y_data[end_index], end_index - start_index + 1)

    for i, y_val in zip(range(start_index, end_index + 1), interpolated_values):
        y_data[i] = y_val

    return jsonify({'y': y_data})

@app.route('/get_original_data', methods=['GET'])
def get_original_data():
    #history.append(y_data.copy())
    global x_data, y_data
    data_dict = next(item for item in model.get_shape_functions_as_dict() if item['name'] == 's1')
    x_data = data_dict['x'].tolist()
    y_data = data_dict['y'].tolist()
    return jsonify({'x': x_data, 'y': y_data})

@app.route('/undo_last_change', methods=['GET'])
def undo_last_change():
    if len(history) > 1:
        # Pop the current data, and set y_data to the previous one
        history.pop()
        global y_data
        y_data = history[-1]
        return jsonify({'y': y_data})
    else:
        return jsonify({'error': 'No more changes to undo'}), 400



if __name__ == '__main__':
    app.run(debug=True)