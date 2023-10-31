from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from igann import IGANN
import numpy as np
import aif360
from aif360.sklearn.datasets.utils import standardize_dataset
#test = standardize_dataset()
compas = aif360.sklearn.datasets.fetch_compas()


X, y = load_diabetes(return_X_y=True, as_frame=True)
scaler = StandardScaler()
X_names = X.columns

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=X_names)
X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')

y = (y - y.mean()) / y.std()

model = IGANN(task='regression')
model.fit(X, y)
#data_dict = model.get_shape_functions_as_dict()
#print(data_dict[0]['x'].tolist())
#model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])




app = Flask(__name__)
data_dict = next(item for item in model.get_shape_functions_as_dict() if item['name'] == 's1')

# Extract x and y data and convert to lists
x_data = data_dict['x'].tolist()
y_data = data_dict['y'].tolist()
#x_data = data_dict[0]['x'].tolist()
#y_data = data_dict[0]['y'].tolist()
#x_data = list(range(10))
#y_data = [1, 4, 2, 2.5, 2, 5, 4, 6, 5, 7]

history = [y_data.copy()]


@app.route('/')
def index():
    return render_template('index.html', x_data=x_data, y_data=y_data)

@app.route('/setConstantValue', methods=['POST'])
def setConstantValue():
    history.append(y_data.copy())
    data = request.json
    x1, x2 = data['x1'], data['x2']
    new_y = float(data['new_y'])
    for i, x in enumerate(x_data):
        if x1 <= x <= x2:
            y_data[i] = new_y
    return jsonify({'y': y_data})


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
