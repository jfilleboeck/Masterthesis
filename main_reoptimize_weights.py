import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from Dashboard.data_preprocessing import load_and_preprocess_data
from Dashboard.model_adapter import ModelAdapter

if __name__ == "__main__":
    print("Running main script")

    # Load and split the data
    X_train, X_test, y_train, y_test, task = load_and_preprocess_data("titanic")
    #X_train, X_test, y_train, y_test, task = load_and_preprocess_data()
    adapter = ModelAdapter(task)
    adapter.fit(X_train, y_train)

    # Calculate and print mean squared error
    y_train_pred = adapter.predict(X_train)
    y_test_pred = adapter.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"MSE Train: {mse_train}, MSE Test: {mse_test}")

    # Create a directory for plots
    # plot_dir = "feature_plots"
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    #
    # # List of features to iterate
    #features_to_change = ['bmi', 'bp']
    features_to_change = ['Age']

    # Load feature data
    shape_functions_dict = adapter.model.get_shape_functions_as_dict()
    #adapter.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])
    adapter.plot_single(plot_by_list=['Age'])
    #adapter.plot_single(plot_by_list=['bmi', 'bp'])
    # this part is already given in the flask application
    feature_current_state = {}
    for feature in shape_functions_dict:
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
                y_values = np.array(feature_current_state[name])
            else:
                y_values = np.where(np.array(feature_current_state[name]) > 0, 0, feature_current_state[name])
                #y_values = feature_current_state[name]
        else:
            # Use the original 'y' values from shape_functions_dict if there is no user change
            y_values = feature['y']

        if feature['datatype'] == 'numerical':
            updated_data[name] = {'x': x_values, 'y': y_values.tolist(), 'datatype': 'numerical'}
        else:
            updated_data[name] = {'x': x_values, 'y': np.array([-0.5]), 'datatype': 'categorical'}

    # ändern, wenn es in Wesite Code kommt: Nur die Änderungen von features_to_change in feature_current_state übernehmen
    # updated_data == feature-current_state; anpassen für kategorische Werte
    # Als erstes möchte ich eine Liste von features to change übergeben

    adapter = adapter.adapt(features_to_change, updated_data, "retrain_feature", X_train, y_train)

    y_train_pred = adapter.predict(X_train)
    y_test_pred = adapter.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"MSE Train: {mse_train}, MSE Test: {mse_test}")
    #adapter.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])
    adapter.plot_single(plot_by_list=['Age'])
    #adapter.plot_single(plot_by_list=['bmi', 'bp'])


    # selected_features = features_to_change
    #for feature in features_to_change:

    #feature_data = next((item for item in shape_functions_dict if item['name'] == selected_feature), None)
    #x_data = feature_data['x'].tolist()



    # Später dann durch features_to changes durchiterieren und stets nur eines verändern


    # for feature_to_change in features_to_change:
    #     # Get the feature data
    #     dict_data = next(item for item in model.get_shape_functions_as_dict() if item['name'] == feature_to_change)
    #     x = dict_data['x']
    #     y_data = dict_data['y']
    #     y = [0 if x < 0 else x for x in y_data]
    #
    #     # Plot and save the parabola
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x, y, label=f'y = {feature_to_change}')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.title(f'Plot of {feature_to_change}')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plot_dir, f"{feature_to_change}_parabola.png"))
    #     plt.close()
    #
    #     # Update and re-optimize the model
    #     model = model.adapt(selected_features=feature_to_change, updated_data=y, X_train=X_train, y_train=y_train, method="retrain_feature")
    #
    #     # Plot and save the model's new predictions
    #     model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])
    #     plt.savefig(os.path.join(plot_dir, f"{feature_to_change}_model_plot.png"))
    #     plt.close()
    #
    #     # Calculate and print mean squared error
    #     y_train_pred = model.predict(X_train)
    #     y_test_pred = model.predict(X_test)
    #     mse_train = mean_squared_error(y_train, y_train_pred)
    #     mse_test = mean_squared_error(y_test, y_test_pred)
    #     print(f"{feature_to_change} - MSE Train: {mse_train}, MSE Test: {mse_test}")
