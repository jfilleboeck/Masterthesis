import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from Dashboard.data_preprocessing import load_and_preprocess_data
from Dashboard.model_adapter import ModelAdapter
import torch
import seaborn as sns




def plot_single(adapter, perfect_plot = None, plot_by_list=None, show_n=5, scaler_dict=None, max_cat_plotted=4):
    """
    This function plots the most important shape functions.
    Parameters:
    show_n: the number of shape functions that should be plotted.
    scaler_dict: dictionary that maps every numerical feature to the respective (sklearn) scaler.
                 scaler_dict[num_feature_name].inverse_transform(...) is called if scaler_dict is not None
    """
    shape_functions = adapter.get_shape_functions_as_dict()
    if plot_by_list is None:
        top_k = [
                    d
                    for d in sorted(
                shape_functions, reverse=True, key=lambda x: x["avg_effect"]
            )
                ][:show_n]
        show_n = min(show_n, len(top_k))
    else:
        top_k = [
            d
            for d in sorted(
                shape_functions, reverse=True, key=lambda x: x["avg_effect"]
            )
        ]
        show_n = len(plot_by_list)

    plt.close(fig="Shape functions")
    fig, axs = plt.subplots(
        2,
        show_n,
        figsize=(14, 4),
        gridspec_kw={"height_ratios": [5, 1]},
        num="Shape functions",
    )
    plt.subplots_adjust(wspace=0.4)

    i = 0
    for d in top_k:
        if plot_by_list is not None and d["name"] not in plot_by_list:
            continue
        if scaler_dict:
            d["x"] = (
                scaler_dict[d["name"]]
                .inverse_transform(d["x"].reshape(-1, 1))
                .squeeze()
            )
        if d["datatype"] == "categorical":
            if show_n == 1:
                d["y"] = np.array(d["y"])
                d["x"] = np.array(d["x"])
                hist_items = [d["hist"][0][0].item()]
                hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                idxs_to_plot = np.argpartition(
                    np.abs(d["y"]),
                    -(len(d["y"]) - 1)
                    if len(d["y"]) <= (max_cat_plotted - 1)
                    else -(max_cat_plotted - 1),
                )[-(max_cat_plotted - 1):]
                y_to_plot = d["y"][idxs_to_plot]
                x_to_plot = d["x"][idxs_to_plot].tolist()
                hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                if len(d["x"]) > max_cat_plotted - 1:
                    # other classes:
                    if "others" in x_to_plot:
                        x_to_plot.append(
                            "others_" + str(np.random.randint(0, 999))
                        )  # others or else seem like plausible variable names
                    else:
                        x_to_plot.append("others")
                    y_to_plot = np.append(y_to_plot.flatten(), [[0]]).reshape(
                        max_cat_plotted,
                    )
                    hist_items_to_plot.append(
                        np.sum(
                            [
                                hist_items[i]
                                for i in range(len(hist_items))
                                if i not in idxs_to_plot
                            ]
                        )
                    )

                axs[0].bar(
                    x=x_to_plot, height=y_to_plot, width=0.5, color="darkblue"
                )
                axs[1].bar(
                    x=x_to_plot,
                    height=hist_items_to_plot,
                    width=1,
                    color="darkblue",
                )

                axs[0].set_title(
                    "{}:\n{:.2f}%".format(
                        adapter._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0].grid()
            else:
                d["y"] = np.array(d["y"])
                d["x"] = np.array(d["x"])
                hist_items = [d["hist"][0][0].item()]
                hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                idxs_to_plot = np.argpartition(
                    np.abs(d["y"]),
                    -(len(d["y"]) - 1)
                    if len(d["y"]) <= (max_cat_plotted - 1)
                    else -(max_cat_plotted - 1),
                )[-(max_cat_plotted - 1):]
                y_to_plot = d["y"][idxs_to_plot]
                x_to_plot = d["x"][idxs_to_plot].tolist()
                hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                if len(d["x"]) > max_cat_plotted - 1:
                    # other classes:
                    if "others" in x_to_plot:
                        x_to_plot.append(
                            "others_" + str(np.random.randint(0, 999))
                        )  # others or else seem like plausible variable names
                    else:
                        x_to_plot.append("others")
                    y_to_plot = np.append(y_to_plot.flatten(), [[0]]).reshape(
                        max_cat_plotted,
                    )
                    hist_items_to_plot.append(
                        np.sum(
                            [
                                hist_items[i]
                                for i in range(len(hist_items))
                                if i not in idxs_to_plot
                            ]
                        )
                    )

                axs[0][i].bar(
                    x=x_to_plot, height=y_to_plot, width=0.5, color="darkblue"
                )
                axs[1][i].bar(
                    x=x_to_plot,
                    height=hist_items_to_plot,
                    width=1,
                    color="darkblue",
                )

                axs[0][i].set_title(
                    "{}:\n{:.2f}%".format(
                        adapter._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0][i].grid()

        else:
            if show_n == 1:
                g = sns.lineplot(
                    x=d["x"], y=d["y"], ax=axs[0], linewidth=2, color="darkblue"
                )
                g.axhline(y=0, color="grey", linestyle="--")
                axs[1].bar(
                    d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                )
                axs[0].set_title(
                    "{}:\n{:.2f}%".format(
                        adapter._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0].grid()
            else:
                g = sns.lineplot(
                    x=d["x"], y=d["y"], ax=axs[0][i], linewidth=2, color="darkblue"
                )
                g.axhline(y=0, color="grey", linestyle="--")
                axs[1][i].bar(
                    d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                )
                axs[0][i].set_title(
                    "{}:\n{:.2f}%".format(
                        adapter._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0][i].grid()
        if perfect_plot is not None and plot_by_list is not None:
            if d["name"] in perfect_plot and d["name"] in plot_by_list:
                # Retrieve the perfect_plot data for this feature
                perfect_data = perfect_plot[d["name"]]
                if d["datatype"] == "categorical":
                    # For categorical data, use a red line at the top of the bars
                    for idx, cat in enumerate(d["x"]):
                        if cat in perfect_data["x"]:
                            perfect_idx = perfect_data["x"].index(cat)
                            y_val = perfect_data["y"][perfect_idx]
                            # Plot a small red line at the top of each bar
                            if show_n == 1:
                                axs[0].plot([cat, cat], [0, y_val], color="red", linewidth=1)
                            else:
                                axs[0][i].plot([cat, cat], [0, y_val], color="red", linewidth=1)
                else:
                    # For numerical data, plot a red line over the existing plot
                    if show_n == 1:
                        axs[0].plot(perfect_data["x"], perfect_data["y"], color="red", linewidth=1)
                    else:
                        axs[0][i].plot(perfect_data["x"], perfect_data["y"], color="red", linewidth=1)

        i += 1

    if show_n == 1:
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
    else:
        for i in range(show_n):
            axs[1][i].get_xaxis().set_visible(False)
            axs[1][i].get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    print("Running main script")
    #"Variations_ELM_Scale_10", "Variations_ElM_Scale_01",
    #experiments = ["Original", "Variations_ELM_Scale", "Synthetic_Data_20"]
    experiment = "Original"
    linear_features = {"diabetes": ("regression", ["bmi", "bp", "s1", "s2", "s3", "s4"]),
                        "titanic": ("classification", ["age", "fare"]),
                        }
    # linear_features = {"diabetes": ("regression", ["bmi", "bp", "s1", "s2", "s3", "s4"]),
    #                     "titanic": ("classification", ["age", "fare"]),
    #                     "adult": ("classification", ["education-num", "hours-per-week", "fnlwgt"])}


    folder_path = os.path.join(os.getcwd(), experiment)
    os.makedirs(folder_path, exist_ok=True)
    result = pd.DataFrame(columns=["Dataset", "Feature", "Median (negative/positive)", "MSE median",
                                   "Extreme (negative/positive)", "MSE extreme", "MSE doubling"])
    simulated_user_adjustments = ["constant_median", "constant_extreme", "doubling"]
    for dataset, (task, features_to_change) in linear_features.items():
        X_train, X_test, y_train, y_test, task = load_and_preprocess_data(dataset)
        for feature_to_change in features_to_change:
            # Train the initial model, calculate initial predictions & mean/extreme
            adapter = ModelAdapter(task)
            adapter.fit(X_train, y_train)
            y_train_pred = adapter.predict(X_train)
            y_test_pred = adapter.predict(X_test)
            if task == "regression":
                initial_metric_train = mean_squared_error(y_train, y_train_pred)
                initial_metric_test = mean_squared_error(y_test, y_test_pred)
            else:
                initial_metric_train = f1_score(y_train, y_train_pred, average='weighted')
                initial_metric_test = f1_score(y_test, y_test_pred, average='weighted')
            shape_functions_dict = adapter.model.get_shape_functions_as_dict()
            feature_current_state = {}
            median_negative = median_positive = mse_median = most_negative = most_positive = mse_extreme = mse_doubling = None

            updated_data = {}

            for adjustment in simulated_user_adjustments:

                for feature in shape_functions_dict:
                    name = feature['name']
                    x_values = feature['x']
                    feature_current_state[name] = feature['y']
                    if name in features_to_change:
                        # Simulate user input
                        y_values = np.array(feature_current_state[name])
                        # saved for output table
                        median_negative = np.median(y_values[y_values < 0])
                        median_positive = np.median(y_values[y_values > 0])
                        most_negative = np.min(y_values[y_values < 0])
                        most_positive = np.max(y_values[y_values > 0])

                        if adjustment == "constant_median":
                            y_values[y_values < 0] = median_negative
                            y_values[y_values > 0] = median_positive
                        elif adjustment == "constant_extreme":
                            y_values[y_values < 0] = most_negative
                            y_values[y_values > 0] = most_positive
                        else:
                            y_values_modified = np.where(y_values != 0, 2*y_values, y_values)

                        synthetic_data_points_nr = 0

                        if synthetic_data_points_nr > 0:
                            transformed_y_values = y_values
                            new_x_values = []
                            new_y_values = []
                            for i in range(len(x_values) - 1):
                                new_x_values.append(x_values[i])
                                new_y_values.append(transformed_y_values[i])
                                # Calculate steps for synthetic points
                                x_step = (x_values[i + 1] - x_values[i]) / (synthetic_data_points_nr + 1)
                                y_step = (transformed_y_values[i + 1] - transformed_y_values[i]) / (
                                            synthetic_data_points_nr + 1)
                                #
                                for j in range(1, synthetic_data_points_nr + 1):
                                    synthetic_x = x_values[i] + j * x_step
                                    synthetic_y = transformed_y_values[i] + j * y_step if transformed_y_values[
                                                                                              i] != -10 else -10
                                    new_x_values.append(synthetic_x)
                                    new_y_values.append(synthetic_y)
                            #
                            # Add the last original point
                            new_x_values.append(x_values[-1])
                            new_y_values.append(transformed_y_values[-1])
                            x_values = new_x_values
                            y_values = new_y_values
                        else:
                            print("No synthetic points")
                        #                       'datatype': 'numerical'}
                    else:
                        # Use the original 'y' values from shape_functions_dict if there is no user change
                        y_values = feature['y']

                    if feature['datatype'] == 'numerical':
                        updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'numerical'}
                        # print(updated_data)
                    else:
                        updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'categorical'}

                    # neu trainieren
                adapter.adapt([feature_to_change], updated_data, "feature_retraining")
                #adapter.predict(updated_data[])
                adjusted_shape_functions = adapter.get_shape_functions_as_dict()
                y_optimal = updated_data[feature_to_change]['y']
                y_hat = adjusted_shape_functions[adapter.model.feature_names.index(feature_to_change)]['y']
                mse_change = mean_squared_error(y_optimal, y_hat)
                if adjustment == "constant_median":
                    mse_median = mse_change
                elif adjustment == "constant_extreme":
                    mse_extreme = mse_change
                else:
                    mse_median = mse_change

                 # plot Methode anpassen, dass es auserdem ein Feature Original gibt, bei dem kein roter Strich eingezeichnet wird
                 # außerdem soll unterdem Plot der MSE/F1 train/test angezeigt werden



            # Rows erst am ende hinzufügen
            row = {"Dataset": dataset,
                   "Feature": feature_to_change,
                   "Median (negative/positive)": (median_negative, median_positive),
                   "MSE median": mse_median,
                   "Extreme (negative/positive)": (most_negative, most_positive),
                   "MSE extreme": mse_extreme,
                   "MSE doubling": mse_doubling}
            result = result._append(row, ignore_index=True)

    print(result.to_string())


    # Load feature data





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



