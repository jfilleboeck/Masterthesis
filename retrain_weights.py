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
    #features_to_change = ['age']
    features_to_change = ['Sex', 'Fare']
    # Load feature data
    shape_functions_dict = adapter.model.get_shape_functions_as_dict()
    #adapter.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])
    adapter.plot_single(plot_by_list=features_to_change)
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
                y_values = np.where(np.array(feature_current_state[name]) < 0, -0.5, feature_current_state[name])
                #y_values = feature_current_state[name]
        else:
            # Use the original 'y' values from shape_functions_dict if there is no user change
            y_values = feature['y']

        if feature['datatype'] == 'numerical':
            updated_data[name] = {'x': x_values, 'y': y_values.tolist(), 'datatype': 'numerical'}
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
    #self.feature_names.index(feature)
    i = 7
    pred = adapter.init_classifier.coef_[0, i] * np.array([0, 1])
    print(pred)
    print(adapter.feature_names)
    pred = 0

    for regressor, boost_rate in zip(adapter.regressors, adapter.boosting_rates):
        pred += (
            boost_rate
            * regressor.predict_single((torch.tensor(-1, dtype=torch.float)).reshape(-1, 1), i).squeeze()
        ).cpu()
    print(pred)
