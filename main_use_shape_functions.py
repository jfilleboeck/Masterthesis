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
    X_train, X_test, y_train, y_test, task = load_and_preprocess_data()
    model = ModelAdapter(task)
    model.fit(X_train, y_train)

    # Create a directory for plots
    plot_dir = "feature_plots_shape"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # List of features to iterate
    features_to_change = ['age', 'bmi', 'bp', 's1', 's2']
    selected_features = []
    updated_data = {}
    for feature_to_change in features_to_change:

        # Get the feature data
        dict_data = next(item for item in model.get_shape_functions_as_dict() if item['name'] == feature_to_change)
        x = dict_data['x']
        y_data = dict_data['y']
        y = [0 if x < 0 else x for x in y_data]

        selected_features.append(feature_to_change)
        updated_data[feature_to_change] = y
        # Plot and save the parabola
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=f'y = {feature_to_change}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Plot of {feature_to_change}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"{feature_to_change}_parabola.png"))
        plt.close()

        # Update and re-optimize the model
        model = ModelAdapter(task)
        model.fit(X_train, y_train)
        model = model.adapt(features_to_change=selected_features, updated_data=updated_data, X_train=X_train, y_train=y_train, method="spline")

        # Plot and save the model's new predictions
        model.plot_single(plot_by_list=['age', 'bmi', 'bp', 's1', 's2'])
        #plt.savefig(os.path.join(plot_dir, f"{feature_to_change}_model_plot.png"))
        plt.close()

        # Calculate and print mean squared error
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        print(f"{feature_to_change} - MSE Train: {mse_train}, MSE Test: {mse_test}")
