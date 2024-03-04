import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from Dashboard.data_preprocessing import load_and_preprocess_data
from Dashboard.model_adapter import ModelAdapter
import torch

if __name__ == "__main__":
    print("Running main script")

    # Load and split the data    X_train, X_test, y_train, y_test, task = load_and_preprocess_data("iris")
    #X_train, X_test, y_train, y_test, task = load_and_preprocess_data("iris")
    X_train, X_test, y_train, y_test, task = load_and_preprocess_data("titanic")
    #X_train, X_test, y_train, y_test, task = load_and_preprocess_data()
   # X_train, X_test, y_train, y_test, task = load_and_preprocess_data("bike")

    #X_train, X_test, y_train, y_test, task = load_and_preprocess_data("titanic")

    adapter = ModelAdapter(task)
    adapter.fit(X_train, y_train)

    # Calculate and print mean squared error
    y_train_pred = adapter.predict(X_train)
    y_test_pred = adapter.predict(X_test)

    if task == "regression":
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        print(f"MSE Train: {mse_train}, MSE Test: {mse_test}")
    else:
        f1_train = f1_score(y_train, y_train_pred, average='weighted')
        f1_test = f1_score(y_test, y_test_pred, average='weighted')
        print(f"Train F1 Score: {f1_train}, Test F1 Score: {f1_test}")



    #features_to_change = ['sepal length (cm)', 'sepal width (cm)',  'petal length (cm)', 'petal width (cm)']
    #features_to_change = ['sepal length (cm)', 'sepal width (cm)']
    #features_to_change = ['petal length (cm)', 'petal width (cm)']

    #features_to_change = ['sepal width (cm)']
    #features_to_change = ['temp']
    #features_to_change = ['bmi', 'bp', 'sex']
    features_to_change = ['Age']
    #features_to_change = ['education_num', 'workclass', 'marital-status', 'capital-loss']
    # Load feature data
    shape_functions_dict = adapter.model.get_shape_functions_as_dict()
    #adapter.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])
    #adapter.plot_single(plot_by_list=features_to_change)
    adapter.model.plot_single(show_n=10)
    print(adapter.model)
    # this part is already given in the flask application
    feature_current_state = {}
    for feature in (
            shape_functions_dict):
        name = feature['name']
        y_value = feature['y']
        feature_current_state[name] = y_value

    updated_data = {}


    for feature in shape_functions_dict:
        name = feature['name']
        x_values = feature['x']
        if name in features_to_change:
            # Simulate user input (by setting negative values to 0); in flask app just use feature_current_state
            if feature["datatype"] == "categorical":
                user_input = -1.5
                feature_current_state[name][0] = -1.5
                y_values = np.array(feature_current_state[name])
                #y_values = {'x': x_values, 'y': np.where(np.array(feature_current_state[name]) < 0,
                #                                         -2, np.array(feature_current_state[name])), 'datatype': 'categorical'}
                #y_values = np.array(feature_current_state[name])
            else:
                #y_values = np.where(np.array(feature_current_state[name]) < 0, -10, feature_current_state[name])
                y_values = np.array(feature_current_state[name])
                #y_values = np.where(y_values < 0, -3, y_values)
                #updated_data[name] = {'x': x_values.tolist(), 'y': adjusted_y_values.tolist(),
                #                      'datatype': 'numerical'}
                synthetic_data_points_nr = 0
                new_x_values = []
                new_y_values = []
                #transformed_y_values = np.where(y_values < 0.8, 0.9, y_values)
                transformed_y_values = np.where(y_values > -5, 3, y_values)
                if synthetic_data_points_nr > 0:
                    for i in range(len(x_values) - 1):
                        new_x_values.append(x_values[i])
                        new_y_values.append(transformed_y_values[i])
                        # Calculate steps for synthetic points
                        x_step = (x_values[i + 1] - x_values[i]) / (synthetic_data_points_nr + 1)
                        y_step = (transformed_y_values[i + 1] - transformed_y_values[i]) / (synthetic_data_points_nr + 1)
                        #
                        for j in range(1, synthetic_data_points_nr + 1):
                            synthetic_x = x_values[i] + j * x_step
                            synthetic_y = transformed_y_values[i] + j * y_step if transformed_y_values[i] != -10 else -10
                            new_x_values.append(synthetic_x)
                            new_y_values.append(synthetic_y)
                #
                #    # Don't forget to add the last original point
                    new_x_values.append(x_values[-1])
                    new_y_values.append(transformed_y_values[-1])
                    x_values = new_x_values
                    y_values= new_y_values
                else:
                     y_values = transformed_y_values
                     print("No synthetic points")
                #                       'datatype': 'numerical'}
        else:
            # Use the original 'y' values from shape_functions_dict if there is no user change
            y_values = feature['y']

        if feature['datatype'] == 'numerical':
            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'numerical'}
        else:
            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'categorical'}


    # ändern, wenn es in Wesite Code kommt: Nur die Änderungen von features_to_change in feature_current_state übernehmen
    # updated_data == feature-current_state; anpassen für kategorische Werte
    # Als erstes möchte ich eine Liste von features to change übergeben

    adapter = adapter.adapt(features_to_change, updated_data, "retrain_feature", X_train, y_train)

    y_train_pred = adapter.predict(X_train)
    y_test_pred = adapter.predict(X_test)
    if task == "regression":
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        print(f"MSE Train: {mse_train}, MSE Test: {mse_test}")
    else:
        f1_train = f1_score(y_train, y_train_pred, average='weighted')
        f1_test = f1_score(y_test, y_test_pred, average='weighted')
        print(f"Train F1 Score: {f1_train}, Test F1 Score: {f1_test}")

    adapter.plot_single(plot_by_list=features_to_change)
    print(adapter.feature_names)
    x = torch.tensor([-2.0177, -2.0005, -1.9949, -1.9894, -1.9832, -1.9777, -1.9086, -1.8396,
     -1.7705, -1.7015, -1.6324, -1.5634, -1.4943, -1.4253, -1.3562, -1.2872,
     -1.2181, -1.1491, -1.0800, -1.0455, -1.0110, -0.9420, -0.8729, -0.8039,
     -0.7348, -0.6658, -0.6312, -0.5967, -0.5277, -0.4586, -0.4241, -0.3896,
     -0.3205, -0.2515, -0.1824, -0.1134, -0.0789, -0.0443, 0.0247, 0.0592,
     0.0938, 0.1628, 0.1973, 0.2319, 0.3009, 0.3700, 0.4390, 0.5080,
     0.5771, 0.6461, 0.7152, 0.7497, 0.7842, 0.8533, 0.9223, 0.9914,
     1.0604, 1.0950, 1.1295, 1.1985, 1.2676, 1.3366, 1.4057, 1.4747,
     1.5438, 1.6819, 1.7509, 1.7854, 1.8200, 1.8890, 1.9580, 2.0271,
     2.0961, 2.1652, 2.2342, 2.3033, 2.3723, 2.4414, 2.5104, 2.7866,
     2.8557, 3.0628, 3.4771])
    i = 0
    #[-2, -1.7, -1.3, -1, 0 , 1, 2]
    print("Intercept: ")
    print(adapter.init_classifier.intercept_)
    pred = adapter.init_classifier.coef_[0, i] * np.array(x)
    print("Prediction Init Classifier: ")
    print(pred)
    # print(adapter.feature_names)

    #pred_new = np.array([0, 0, 0, 0, 0])
    pred_new = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float)

    #pred_new = pred.tolist()
    for regressor, boost_rate in zip(adapter.regressors, adapter.boosting_rates):
        pred_new += (
            boost_rate
            * regressor.predict_single((torch.tensor([-2, -1.7, -1.3, -1, 0 , 1, 2], dtype=torch.float)).reshape(-1, 1), i).squeeze()
        ).cpu()
    print("Prediction Regressoren: ")
    print(pred_new)

    # for regressor, boost_rate in zip(adapter.regressors, adapter.boosting_rates):
    #     pred_new += (
    #         boost_rate
    #         * regressor.predict_single((torch.tensor([-2, -1, 0 , 1, 2], dtype=torch.float)).reshape(-1, 1), i).squeeze()
    #     ).cpu()
    # print("Prediction Regressoren: ")
    # print(pred_new)
