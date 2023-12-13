from igann import IGANN
import torch
import numpy as np
from igann.igann import ELM_Regressor
from scipy.interpolate import CubicSpline


class ModelAdapter():
    # This class contains all the methods, which the backend requires
    # It distributes the methods to the correct method of the ML models
    # using object composition

    def __init__(self, model="IGANN"):
        self.model_name = model
        if self.model_name == "IGANN":
            self.model = IGANN(task='regression')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def adapt(self,  selected_features, updated_data, method, X_train, y_train):
        if self.model_name == "IGANN":
            self.model = IGANNAdapter(task='regression')
            if method == "reoptimize_weights":
                self.model.reoptimize_weights(selected_features, updated_data, X_train, y_train)
            if method == "spline":
                self.model.update_model(selected_features, updated_data, X_train, y_train)

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

    def reoptimize_weights(self, selected_feature, updated_data, X_train, y_train):
        # set weights of selected features to 0
        self.fit(X_train, y_train)
        # Fit the model with adjusted regressors
        i = self.feature_names.index(selected_feature)
        shape_information = next(item for item in self.get_shape_functions_as_dict() if item['name'] == selected_feature)
        x = torch.tensor(shape_information['x'], dtype=torch.float32)
        y_hat = self.init_classifier.coef_[i] * x.numpy() + self.init_classifier.intercept_
        # y = training targets, adjusted for certain x values
        y = torch.tensor(updated_data, dtype=torch.float32) - y_hat
        # TODO: Y_VAL ?
        for counter, regressor in enumerate(self.regressors):
            hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
            y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(
                y, y_hat
            )
            new_regressor = ELM_Regressor(
                n_input=1,
                # TODO: Für kategorische Werte, muss n.categorical_cols angepasst werden
                n_categorical_cols=0,
                n_hid=self.n_hid,
                seed=counter,
                elm_scale=self.elm_scale,
                elm_alpha=self.elm_alpha,
                act=self.act,
                device=self.device,
            )

            X_hid = new_regressor.fit(
                x.reshape(-1, 1),
                y_tilde,
                torch.sqrt(torch.tensor(0.5).to(self.device))
                * self.boost_rate
                * hessian_train_sqrt[:, None],
            )
            # Make a prediction of the ELM for the update of y_hat
            train_regressor_pred = new_regressor.predict(X_hid, hidden=True).squeeze()

            # Update the prediction for training and validation data
            y_hat = torch.tensor(y_hat, dtype=torch.float32)
            y_hat += self.boost_rate * train_regressor_pred

            y_hat = self._clip_p(y_hat)
            # TODO: Ergebnisse in Listen aktualisieren
            print(i)
            print(self.n_hid)
            regressor.hidden_mat[i, i * self.n_hid: (i + 1) * self.n_hid] = new_regressor.hidden_mat.squeeze()
            regressor.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid] = new_regressor.output_model.coef_

    def update_model_with_spline(self, selected_features, updated_data, X_train, y_train):
        self.fit(X_train, y_train)
        # create cubic spline
        spline_functions = {}
        selected_features_i = []

        for feature in selected_features:
            for feature_data in self.get_shape_functions_as_dict():
                if feature_data['name'] == feature:
                    feature_index = self.feature_names.index(feature)
                    x_data = feature_data['x']
                    y_data = updated_data[feature]  # New y data
                    spline = CubicSpline(x_data, y_data)
                    spline_functions[feature_index] = spline
                    selected_features_i.append(feature_index)
                    break

        # Copy ELM Regressor
        new_regressors = []
        # Create custom Regressor
        for index, regressor in enumerate(self.regressors):
            new_regressor=ELM_Regressor_Spline(
                n_input=regressor.n_input,
                n_categorical_cols=regressor.n_categorical_cols,
                n_hid=regressor.n_hid,
                # seed is not an instance attribute in IGANN code
                seed=getattr(regressor, 'seed', 0),
                elm_scale=regressor.elm_scale,
                elm_alpha=regressor.elm_alpha,
                act=regressor.act,
                device=regressor.device,
                selected_features_i=selected_features_i,
                spline_functions=spline_functions
            )
            # Copying all attributes from the old regressor to the new regressor
            for attr in vars(regressor):
                setattr(new_regressor, attr, getattr(regressor, attr))
            new_regressor.spline_functions = spline_functions
            new_regressors.append(new_regressor)
        self.regressors = new_regressors




class ELM_Regressor_Spline(ELM_Regressor):
    def __init__(self, selected_features_i, spline_functions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_features_i = selected_features_i
        self.spline_functions = spline_functions

    def predict(self, X, hidden=False):
        """
        This function makes a full prediction with the model for a given input X.
        """
        if hidden:
            X_hid = X
        else:
            X_hid = self.get_hidden_values(X)

        # Set the coefficients for selected features to 0
        for i in self.selected_features_i:
            self.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid] = 0
        # Use the modified coefficients for prediction
        out = X_hid @ self.output_model.coef_
        # Add spline function values for selected features
        for i in self.selected_features_i:
            spline = self.spline_functions[i]
            spline_values = spline(X[:, i])
            out += spline_values

        if isinstance(spline_values, np.ndarray):
            spline_values = torch.from_numpy(spline_values).float()

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
        if i in self.selected_features_i:
            # If the feature is selected, use the spline function for prediction
            self.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid] = 0
            spline = self.spline_functions[i]
            out = spline(x)

            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out).float()
        else:
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

        return out.squeeze()