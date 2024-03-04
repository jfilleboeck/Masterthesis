from igann import IGANN
import torch
import numpy as np
from igann.igann import ELM_Regressor
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt


class ModelAdapter():
    # This class contains all the methods, which the backend requires
    # It distributes the methods to the correct method of the ML models
    # using object composition

    def __init__(self, task, model="IGANN"):
        self.model_name = model
        if self.model_name == "IGANN":
            self.model = IGANNAdapter(task=task)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def adapt(self, features_to_change, updated_data, method, X_train, y_train):
        if self.model_name == "IGANN":
            if method == "retrain_feature":
                self.model.feature_retraining(features_to_change, updated_data, X_train, y_train)
            if method == "spline":
                self.model.spline_interpolation(features_to_change, updated_data, X_train, y_train)
        return self.model

    def plot_single(self, plot_by_list):
        # Check if the underlying model has 'plot_single' method
        if hasattr(self.model, 'plot_single'):
            return self.model.plot_single(plot_by_list)
        else:
            raise NotImplementedError("Plotting is not supported for this model.")

    @property
    def unique(self):
        if self.model_name == "IGANN" and hasattr(self.model, 'unique'):
            return self.model.unique
        else:
            raise AttributeError("Unique attribute not available for this model")

    @property
    def feature_names(self):
        if self.model_name == "IGANN" and hasattr(self.model, 'feature_names'):
            return self.model.feature_names
        else:
            raise AttributeError("Feature names attribute not available for this model")


    def get_shape_functions_as_dict(self):
        if self.model_name == "IGANN":
            return self.model.get_shape_functions_as_dict()

    def predict(self, X):
        return self.model.predict(X)


class IGANNAdapter(IGANN):
    def __init__(self, *args, **kwargs):
        super(IGANNAdapter, self).__init__(*args, **kwargs)
        self.spline_interpolation_run = False

    def encode_categorical_data(self, features_to_change, updated_data):
        features_to_change_extended = []
        updated_data_extended = {}
        for feature in features_to_change:
            # updated data durchlaufen und mit get_shape functions_as dict vergleichen, bei welchen Kategorien etwas geändert wurde
            test_variable = updated_data[feature]
            if updated_data[feature]['datatype'] == "categorical":
                # key = Embarked_Q, value = Embarked; key = Embarked_S, value = Embarked; feature = Embarked
                for key, value in self.dummy_encodings.items():
                    pref_feature_name, category = key.split('_')
                    if value == feature:
                        features_to_change_extended.append(key)
                        # füge hier pro Kategorie ein Key-Value pair updated_data_extended hinzu
                        feature_data = updated_data[feature]
                        # Find the index of the category in the 'x' list
                        if category in feature_data['x']:
                            category_index = feature_data['x'].index(category)
                            # Extract the corresponding value from the 'y' array
                            y_value = feature_data['y'][category_index]
                            updated_data_extended[key] = {
                                'x': [1],
                                'y': np.array([y_value]),
                                'datatype': feature_data['datatype']
                            }
            else:
                features_to_change_extended.append(feature)
                updated_data_extended[feature] = updated_data[feature]
        return features_to_change_extended, updated_data_extended

    def feature_retraining(self, features_to_change, updated_data, X_train, y_train):
        features_to_change, updated_data = self.encode_categorical_data(features_to_change, updated_data)
        # change next line to updated_selected_features
        for feature in features_to_change:
            i = self.feature_names.index(feature)
            x = torch.tensor(updated_data[feature]['x'], dtype=torch.float32)
            #print(feature)
            #print(x.shape)
            y = torch.tensor(updated_data[feature]['y'], dtype=torch.float32)
            # Initialize lists to store the first values for plotting
            y_first_values = []
            train_regressor_pred_first_values = []
            y_hat_first_values = []
            y_tilde_first_values = []
            if self.task == "classification":
                y_hat = torch.tensor(self.init_classifier.coef_[0, i] * x.numpy(), dtype=torch.float64)
            else:
                # + self.init_classifier.intercept_
                y_hat = self.init_classifier.coef_[i] * x.numpy() + self.init_classifier.intercept_
            #print(y)
            #self.task = "regression"
            #y_hat = np.zeros_like(y_hat)
            n_categorical_cols = 1 if updated_data[feature]['datatype'] == 'categorical' else 0
            for counter, regressor in enumerate(self.regressors):
                    # Store the first value of each tensor for plotting
                y_first_values.append(y[0].item())
                y_hat_first_values.append(y_hat[0])

                if updated_data[feature]['datatype'] == 'numerical':
                    if self.task == "classification":
                        #self.task = "regression"
                        #y_tilde = (torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(y, y_hat)).to(
                         #   dtype=torch.float64)
                        y_tilde = torch.sqrt(torch.tensor(0.5)) * (torch.tensor((y - y_hat), dtype=torch.float64))
                        #y_tilde = (torch.tensor((y - y_hat), dtype=torch.float64))
                        self.task = "classification"
                    else:
                         y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(y, y_hat).to(dtype=torch.float32)
                    hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
                    # y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(y, y_hat).to(
                    #     dtype=torch.float64)
                    new_regressor = ELM_Regressor(
                        n_input=1,
                        n_categorical_cols=n_categorical_cols,
                        n_hid=self.n_hid,
                        seed=1,
                        elm_scale=self.elm_scale,
                        # 0.002
                        elm_alpha=1,
                        act=self.act,
                        device=self.device,
                    )
                    if self.task == 'classification':
                        X_hid = new_regressor.fit(
                            x.reshape(-1, 1),
                            # y - y_hat = y_tilde
                            y_tilde,
                            torch.ones_like(torch.sqrt(torch.tensor(0.5).to(self.device))
                            * self.boost_rate
                            * hessian_train_sqrt[:, None])
                        )
                    else:
                        X_hid = new_regressor.fit(
                            x.reshape(-1, 1),
                            # y - y_hat = y_tilde
                            y_tilde,
                            torch.sqrt(torch.tensor(0.5).to(self.device))
                            * self.boost_rate
                            * hessian_train_sqrt[:, None]
                        )
                    # Make a prediction of the ELM for the update of y_hat
                    if self.task == "classification":
                        train_regressor_pred = new_regressor.predict(X_hid.to(dtype=torch.float64), hidden=True).squeeze()
                    else:
                        train_regressor_pred = new_regressor.predict(X_hid, hidden=True).squeeze()

                    train_regressor_pred_first_values.append(train_regressor_pred[0].item())
                    y_tilde_first_values.append(y_tilde[0].item())

                    # Update the prediction for training and validation data
                    y_hat = torch.tensor(y_hat, dtype=torch.float64)
                    y_hat += self.boost_rate * train_regressor_pred
                    y_hat = self._clip_p(y_hat)
                    # replace the weights in the list of original regressors
                    regressor.hidden_mat[i, i * self.n_hid: (i + 1) * self.n_hid] = new_regressor.hidden_mat.squeeze()
                    regressor.output_model.coef_[
                    i * self.n_hid: (i + 1) * self.n_hid] = new_regressor.output_model.coef_
                else:
                    # for categorical features only update the weight
                    # train_regressor_pred = torch.from_numpy(y) * X[:, i]
                    new_weight = y / len(self.regressors) * (1 / self.boosting_rates[counter])
                    regressor.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid] = new_weight

            if len(y_first_values) > 0:  # Ensure there's data to plot
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(y_first_values)), y_first_values, label='y', color='blue')
                plt.scatter(range(len(train_regressor_pred_first_values)), train_regressor_pred_first_values,
                            label='train_regressor_pred', color='red')
                plt.scatter(range(len(y_hat_first_values)), y_hat_first_values, label='y_hat (before update)',
                            color='green')
                plt.scatter(range(len(y_tilde_first_values)), y_tilde_first_values, label='y_tilde', color='orange')

                plt.title('First Value Comparison for First 8 Iterations')
                plt.xlabel('Iteration')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.show()

    def spline_interpolation(self, features_to_change, updated_data, X_train, y_train):
        self.spline_interpolation_run = True
        features_to_change, updated_data = self.encode_categorical_data(features_to_change, updated_data)
        # create cubic spline
        spline_functions = {}
        features_selected_i = []
        for feature in features_to_change:
            feature_index = self.feature_names.index(feature)
            x_data = updated_data[feature]['x']
            y_data = updated_data[feature]['y']
            if updated_data[feature]['datatype'] == 'numerical':
                spline = CubicSpline(x_data, y_data)
            else:
                # for categorical features, spline is the y-value, that the user determined
                spline = y_data
            spline_functions[feature_index] = spline
            features_selected_i.append(feature_index)
            # Required to adjust the predict_single method
        # Copy ELM Regressor
        new_regressors = []
        # Create custom Regressor
        for index, regressor in enumerate(self.regressors):
            new_regressor = ELM_Regressor_Spline(
                n_input=regressor.n_input,
                n_categorical_cols=regressor.n_categorical_cols,
                n_hid=regressor.n_hid,
                # seed is not an instance attribute in IGANN code
                seed=getattr(regressor, 'seed', 0),
                elm_scale=regressor.elm_scale,
                elm_alpha=regressor.elm_alpha,
                act=regressor.act,
                device=regressor.device,
                features_selected_i=features_selected_i,
                spline_functions=spline_functions,
                n_regressors=len(self.regressors),
                boosting_rates=self.boosting_rates,
                index_regressor=index,
                init_classifier=self.init_classifier,
                task=self.task
            )
            # Copying all attributes from the old regressor to the new regressor
            for attr in vars(regressor):
                setattr(new_regressor, attr, getattr(regressor, attr))
            new_regressor.spline_functions = spline_functions
            new_regressors.append(new_regressor)
        self.regressors = new_regressors

    # def _get_pred_of_i(self, i, x_values=None):
    #     if x_values == None:
    #         feat_values = self.unique[i]
    #     else:
    #         feat_values = x_values[i]
    #     if self.task == "classification":
    #         pred = self.init_classifier.coef_[0, i] * feat_values
    #     else:
    #         pred = self.init_classifier.coef_[i] * feat_values
    #     feat_values = feat_values.to(self.device)
    #     if self.task == "classification":
    #         pass
    #         #if self.spline_interpolation_run == True:
    #             #pred = torch.zeros_like(pred)
    #     for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
    #         pred += (
    #             boost_rate
    #             * regressor.predict_single(feat_values.reshape(-1, 1), i).squeeze()
    #         ).cpu()
    #     return feat_values, pred




class ELM_Regressor_Spline(ELM_Regressor):
    def __init__(self, features_selected_i, spline_functions, n_regressors,
                 boosting_rates, index_regressor, init_classifier, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_selected_i = features_selected_i
        self.spline_functions = spline_functions
        self.n_regressors = n_regressors
        self.boosting_rates = boosting_rates
        self.index_regressor = index_regressor
        self.init_classifier = init_classifier
        self.task = task

    def predict(self, X, hidden=False):
        """
        This function makes a full prediction with the model for a given input X.
        """
        if hidden:
            X_hid = X
        else:
            X_hid = self.get_hidden_values(X)

        # Set the coefficients for selected features to 0
        for i in self.features_selected_i:
            if i < self.n_numerical_cols:
                # numerical feature
                self.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid] = 0
            else:
                start_idx = self.n_numerical_cols * self.n_hid + (i - self.n_numerical_cols)
                self.output_model.coef_[start_idx: start_idx + 1] = 0
        # Use the modified coefficients for prediction
        out = X_hid @ self.output_model.coef_
        # Add spline function values for selected features
        for i in self.features_selected_i:
            spline = self.spline_functions[i]
            if self.task == "classification":
                pred_init_classifier = self.init_classifier.coef_[0, i] * X[:, i]
            else:
                pred_init_classifier = self.init_classifier.coef_[i] * X[:, i]
            share_of_init_classifier = np.array(pred_init_classifier / self.n_regressors)
            if i < self.n_numerical_cols:
                prediction = ((spline(X[:, i]) / self.n_regressors) - share_of_init_classifier) * (1/self.boosting_rates[self.index_regressor])
            else:
                # for categorical features, spline is simply the y value, which the user determined
                #prediction = torch.from_numpy(spline)
                user_prediction = torch.from_numpy(self.spline_functions[i]) * X[:, i]
                prediction = ((user_prediction / self.n_regressors) - share_of_init_classifier) * (1 / self.boosting_rates[self.index_regressor])
            out += prediction
        return out

    def predict_single(self, x, i):
        """
        This function computes the partial output of one base function for one feature.
        Note, that the bias term is not used for this prediction.
        Input parameters:
        x: a vector representing the values which are used for feature i
        i: the index of the feature that should be used for the prediction
        """

        # See self.predict for the description - it's almost equivalent.
        x_in = x.reshape(len(x), 1)
        if i not in self.features_selected_i:
            # normal prediction
            x_in = x.reshape(len(x), 1)
            if i < self.n_numerical_cols:
                # numerical feature
                x_in = x_in @ self.hidden_mat[
                              i, i * self.n_hid: (i + 1) * self.n_hid
                              ].unsqueeze(0)
                x_in = self.act(x_in)
                out = x_in @ self.output_model.coef_[
                             i * self.n_hid: (i + 1) * self.n_hid
                             ].unsqueeze(1)
            else:
                # categorical feature
                start_idx = self.n_numerical_cols * self.n_hid + (i - self.n_numerical_cols)
                out = x_in @ self.output_model.coef_[start_idx: start_idx + 1].unsqueeze(1)
            return out
        else:
            if self.task == "classification":
                pred_init_classifier = self.init_classifier.coef_[0, i] * x
            else:
                pred_init_classifier = self.init_classifier.coef_[i] * x
            share_of_init_classifier = np.array(pred_init_classifier / self.n_regressors)
            if i < self.n_numerical_cols:
                # If the feature is selected, use the spline function for prediction
                spline = self.spline_functions[i]
                out = ((spline(x) / self.n_regressors) - share_of_init_classifier) * (1/self.boosting_rates[self.index_regressor])
                if isinstance(out, np.ndarray):
                    out = torch.from_numpy(out).float()
            else:
                prediction = torch.from_numpy(self.spline_functions[i]) * x
                # out = (prediction / self.n_regressors * (1/self.boosting_rates[self.index_regressor])
                #        - pred_init_classifier/self.n_regressors)
                out = ((prediction / self.n_regressors)- share_of_init_classifier) * (1/self.boosting_rates[self.index_regressor])
                if isinstance(out, np.ndarray):
                    out = torch.from_numpy(out).float()
                # if self.index_regressor == 0:
                #     out -= np.array(pred_init_classifier)
            return out.squeeze()

