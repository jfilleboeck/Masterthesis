import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from Dashboard.data_preprocessing import load_and_preprocess_data
from Dashboard.model_adapter import ModelAdapter
import torch
import seaborn as sns
import copy



def plot_single(adapter, shape_functions_for_plotting, target_data_dict, experiment, plot_by_list=None, show_n=5, scaler_dict=None, max_cat_plotted=4):
    """
    This function plots the most important shape functions.
    Parameters:
    show_n: the number of shape functions that should be plotted.
    scaler_dict: dictionary that maps every numerical feature to the respective (sklearn) scaler.
                 scaler_dict[num_feature_name].inverse_transform(...) is called if scaler_dict is not None
    """
    directory_path = os.path.join(os.getcwd(), experiment)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    shape_functions = shape_functions_for_plotting
    # if plot_by_list is None:
    #     top_k = [
    #                 d
    #                 for d in sorted(
    #             shape_functions, reverse=True, key=lambda x: x["avg_effect"]
    #         )
    #             ][:show_n]
    #     show_n = min(show_n, len(top_k))
    # else:
    #     top_k = [
    #         d
    #         for d in sorted(
    #             shape_functions, reverse=True, key=lambda x: x["avg_effect"]
    #         )
    #     ]
    #     show_n = len(plot_by_list)
    if plot_by_list is None:
        top_k = [d for d in shape_functions][:show_n]
        show_n = min(show_n, len(top_k))
    else:
        top_k = [d for d in shape_functions]
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
        file_path = os.path.join(directory_path, f"{d['name']}.png")
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
        if target_data_dict is not None:
            if d["name"] in target_data_dict:
                # Retrieve the perfect_plot data for this feature
                perfect_data = target_data_dict[d["name"]]
                if d["datatype"] == "categorical":
                    # For categorical data, use a red line at the top of the bars
                    for idx, cat in enumerate(d["x"]):
                        if cat in perfect_data["x"]:
                            perfect_idx = perfect_data["x"].index(cat)
                            y_val = perfect_data["y"][perfect_idx]
                            # Plot a small red line at the top of each bar
                            if show_n == 1:
                                axs[0].plot([cat, cat], [0, y_val], color="green", linewidth=1)
                            else:
                                axs[0][i].plot([cat, cat], [0, y_val], color="green", linewidth=1)
                else:
                    # For numerical data, plot a red line over the existing plot
                    if show_n == 1:
                        axs[0].plot(perfect_data["x"], perfect_data["y"], color="green", linewidth=1)
                    else:
                        axs[0][i].plot(perfect_data["x"], perfect_data["y"], color="green", linewidth=1)

        i += 1

    if show_n == 1:
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
    else:
        for i in range(show_n):
            axs[1][i].get_xaxis().set_visible(False)
            axs[1][i].get_yaxis().set_visible(False)
    fig.savefig(file_path)

    plt.show()


if __name__ == "__main__":
    print("Running main script")
    #"Variations_ELM_Scale_10", "Variations_ElM_Scale_01",
    experiments = ["Baseline", "Variations_ELM_Scale", "Variations_ELM_Alpha", "Synthetic_Data_Added"]

    #experiments=["Baseline", "Synthetic_Data_Added"]
    #experiment = "Variations_ELM_Scale"

    #linear_features = {"diabetes": ("regression", ["bmi", "bp", "s1", "s2", "s3", "s4"]),
     #                    "titanic": ("classification", ["age", "fare"]),
      #                   }
    # linear_features = {"diabetes": ("regression", ["bmi", "bp", "age", "s1", "s2", "s3", "s4", "s5", "s6"]),
    #                    "titanic": ("classification", ["age", "fare", "members"]),
    #                    "adult": ("classification", ["education-num", "hours-per-week", "fnlwgt", "age", "capital-gain",
    #                                                 "capital-loss"])}
    linear_features = {"diabetes": ("regression", ["bmi", "bp", "age", "s1", "s2", "s3", "s4", "s5", "s6"]),
                       "titanic": ("classification", ["age", "fare", "members"])}
    #convex_features = {"diabetes": ("regression", ["s6", "age"])}
    mse_median_baseline = {}
    mse_extreme_baseline = {}
    mse_doubling_baseline = {}
    for experiment in experiments:
        # 4 experimente durchlaufen
        # linear_features = {"diabetes": ("regression", ["bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]),
        #                      "titanic": ("classification", ["age", "fare"]),
        #                      }
        folder_path = os.path.join(os.getcwd(), experiment)
        os.makedirs(folder_path, exist_ok=True)
        # result = pd.DataFrame(columns=["Dataset", "Feature", "Median (negative/positive)", "MSE median",
        #                                "Extreme (negative/positive)", "MSE extreme", "MSE doubling", "MSE constant max"])
        result = result_baseline = pd.DataFrame(columns=["Dataset", "Feature", "Range y-values", "MSE median",
                                                        "MSE Extreme", "MSE Doubling"])

        #simulated_user_adjustments = ["set to median", "set to min/max", "double y values", "const. max"]
        simulated_user_adjustments = ["set to median", "set to min/max", "double y values"]
        # [diabetes0.2

        for dataset, (task, features_to_change) in linear_features.items():
            X_train, X_test, y_train, y_test, task = load_and_preprocess_data(dataset)

            for feature_to_change in features_to_change:
                # in jedem Experiment, alle Datensätze und Features durchiterieren
                # pro Feature eine Zeile hinzufügen
                # Train the initial model, calculate initial predictions & mean/extreme
                adapter = ModelAdapter(task)
                adapter.fit(X_train, y_train)
                if experiment == "Variations_ELM_Scale":
                    elm_scale = 10
                else:
                    elm_scale = adapter.model.elm_scale
                if experiment == "Variations_ELM_Alpha":
                    elm_alpha = 0.001
                else:
                    elm_alpha = adapter.model.elm_alpha
                if experiment == "Synthetic_Data_Added":
                    nr_synthetic_data_points = 10
                else:
                    nr_synthetic_data_points = 0
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
                median_negative = median_positive = mse_median = most_negative = most_positive = mse_extreme\
                    = mse_doubling = None

                updated_data = {}
                shape_functions_for_plotting = [(shape_functions_dict[adapter.model.feature_names.index(feature_to_change)])]
                target_data_dict = {}
                for adjustment in simulated_user_adjustments:
                    # pro Benutzeränderung Wert berechnen
                    for feature in shape_functions_dict:
                        name = feature['name']
                        x_values = feature['x']
                        feature_current_state[name] = feature['y']
                        if name == feature_to_change:
                            # Simulate user input
                            y_values = np.array(feature_current_state[name])
                            # saved for output table
                            median_negative = np.median(y_values[y_values < 0])
                            median_positive = np.median(y_values[y_values > 0])
                            most_negative = np.min(y_values[y_values < 0])
                            most_positive = np.max(y_values[y_values > 0])

                            if adjustment == "set to median":
                                y_values[y_values < 0] = median_negative
                                y_values[y_values > 0] = median_positive
                            elif adjustment == "set to min/max":
                                y_values[y_values < 0] = most_negative
                                y_values[y_values > 0] = most_positive
                            elif adjustment == "double y values":
                                y_values = np.where(y_values != 0, 2*y_values, y_values)
                        else:
                            y_values = feature['y']

                        if feature['datatype'] == 'numerical':
                            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'numerical'}
                            # print(updated_data)
                        else:
                            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'categorical'}
                    if experiment == "Synthetic_Data_Added":
                        updated_data_copy = copy.deepcopy(updated_data)
                        # neu trainieren

                    adapter.adapt([feature_to_change], updated_data, "feature_retraining",
                                  (nr_synthetic_data_points, elm_scale, elm_alpha))
                    #adapter.predict(updated_data[])
                    adjusted_shape_functions = adapter.get_shape_functions_as_dict()
                    if experiment == "Synthetic_Data_Added":
                        y_optimal = updated_data_copy[feature_to_change]['y']
                    else:
                        y_optimal = updated_data[feature_to_change]['y']

                    # y_hat muss
                    y_hat = adjusted_shape_functions[adapter.model.feature_names.index(feature_to_change)]['y']

                    mse_change = mean_squared_error(y_optimal, y_hat)
                    if feature_to_change == "bmi" and adjustment == "set to median":

                        if experiment == "Baseline":
                            baseline = mse_change
                        else:
                            out = mse_change - baseline
                            print(out)
                    if experiment == "Baseline":
                        #0.06
                        if adjustment == "set to median":
                            mse_median = mse_change
                            if dataset not in mse_median_baseline:
                                mse_median_baseline[dataset] = {}
                            mse_median_baseline[dataset][feature_to_change] = mse_median
                        elif adjustment == "set to min/max":
                            mse_extreme = mse_change
                            if dataset not in mse_extreme_baseline:
                                mse_extreme_baseline[dataset] = {}
                            mse_extreme_baseline[dataset][feature_to_change] = mse_extreme
                        elif adjustment == "double y values":
                            mse_doubling = mse_change
                            #mse_doubling_baseline = mse_doubling
                            if dataset not in mse_doubling_baseline:
                                mse_doubling_baseline[dataset] = {}
                            mse_doubling_baseline[dataset][feature_to_change] = mse_doubling
                    else:
                        if adjustment == "set to median":
                            mse_median = mse_change - mse_median_baseline[dataset][feature_to_change]
                        elif adjustment == "set to min/max":
                            mse_extreme = mse_change - mse_extreme_baseline[dataset][feature_to_change]
                        elif adjustment == "double y values":
                            mse_doubling = mse_change - mse_doubling_baseline[dataset][feature_to_change]


                     # plot Methode anpassen, dass es auserdem ein Feature Original gibt, bei dem kein roter Strich eingezeichnet wird
                     # außerdem soll unterdem Plot der MSE/F1 train/test angezeigt werden

                    # MITTAGSPAUSE: umändern zur Liste
                    adjusted_shape_function_tmp = adjusted_shape_functions[adapter.model.feature_names.index(feature_to_change)]
                    adjusted_shape_function_tmp['name'] = f"{feature_to_change} - {adjustment}"
                    shape_functions_for_plotting.append(adjusted_shape_function_tmp)


                    datatype = adjusted_shape_functions[adapter.model.feature_names.index(feature_to_change)]['datatype']
                    target_data_dict[f"{feature_to_change} - {adjustment}"] =\
                        {'x': updated_data[feature_to_change]['x'], 'y': updated_data[feature_to_change]['y'],
                         'datatype': updated_data[feature_to_change]['datatype']}



                #plot_single(adapter.model, shape_functions_for_plotting, target_data_dict, experiment)
                # Rows erst am ende hinzufügen
                row = {"Dataset": dataset,
                       "Feature": feature_to_change,
                       #"Median (negative/positive)": (median_negative, median_positive),
                       "Range y-values": (most_negative, most_positive),
                       "MSE median": mse_median,
                       "MSE Extreme": mse_extreme,
                       "MSE Doubling": mse_doubling}
                result = result._append(row, ignore_index=True)
        # auf dieser Ebene, jedes der 4 Experimente durcken
        print(experiment)
        print(result.to_string())


    # Load feature data





    # y_train_pred = adapter.predict(X_train)
    # y_test_pred = adapter.predict(X_test)
    # if task == "regression":
    #     mse_train = mean_squared_error(y_train, y_train_pred)
    #     mse_test = mean_squared_error(y_test, y_test_pred)
    #     print(f"MSE Train: {mse_train}, MSE Test: {mse_test}")
    # else:
    #     f1_train = f1_score(y_train, y_train_pred, average='weighted')
    #     f1_test = f1_score(y_test, y_test_pred, average='weighted')
    #     print(f"Train F1 Score: {f1_train}, Test F1 Score: {f1_test}")



