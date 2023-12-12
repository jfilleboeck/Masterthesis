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
            print("erfolgreich")
            self.model = IGANN(task='regression')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def adapt(self,  selected_features, updated_data, method, X_train, y_train):
        if self.model_name == "IGANN":
            self.model = IGANNAdapter(task='regression')
            self.model.update_model(selected_features, updated_data, method, X_train, y_train)
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


    def update_model(self, selected_features, updated_data, method, X_train, y_train):
        # set weights of selected features to 0
        self.fit(X_train, y_train)
        # add value from spline
        if method=="spline":
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
            # Option 1: Adjust the predict function of the regressor

            # Option 2: Adjust the weights of the regressor


        if method=="reoptimize_weights":
            # Fit the model with adjusted regressors
            # y muss angepasst werden. Wenn in selected features, dann updated_data, sonst nicht
            # y_hat überprüfen

            #X_val, y_val, y_hat_val

            for feature in selected_features:
                print(feature)
                x = X_train[feature]
                y_hat = self.init_classifier.coef_[self.feature_names.index(feature)] * x + self.init_classifier.intercept_
                #y = training targets, adjusted for certain x values
                y = updated_data[feature]
                hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
                y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(
                    y, y_hat
                )
                # y_val? Habe ich nicht
                print(y_tilde)
                for regressor in self.regressors:
                    print(regressor)

        #print(X_train["age"])

    def predict_raw(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        """
        X = self._preprocess_feature_matrix(X, fit_dummies=False).to(self.device)
        X = X[:, self.feature_indizes]

        self.pred_nn = torch.zeros(len(X), dtype=torch.float32).to(self.device)
        for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
                self.pred_nn += boost_rate * regressor.predict(X).squeeze()
        self.pred_nn = self.pred_nn.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        pred = (
            self.pred_nn
            + (self.init_classifier.coef_.astype(np.float32) @ X.transpose()).squeeze()
            + self.init_classifier.intercept_
        )

        return pred


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