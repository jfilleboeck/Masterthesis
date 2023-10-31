class IGANNAdapter():
    def __init__(self, igann_model):
        self.m = igann_model

    def replace_shape_curve(self, feature_name, method, new_x, new_y):
        # create curve via splines
        pass

    def nullify_regressor(self, feature_name):
        pass

    def _add_function(self, function_name):
        # append new shape function to list, so that multiple new
        # shape functions can be executed
        pass

    def predict(self):
        # call default IGANN predict method
        pass
