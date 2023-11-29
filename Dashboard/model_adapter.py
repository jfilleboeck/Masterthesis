from igann import IGANN

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

    def adapt(self,  selected_features, updated_data, X_train, y_train, method="spline"):
        if self.model_name == "IGANN":
            self.model = IGANNAdapter(task='regression')
            if method == "spline":
                self.model.update_model_with_spline(selected_features, updated_data, X_train, y_train)
        return self.model


    def get_shape_functions_as_dict(self):
        if self.model_name == "IGANN":
             return self.model.get_shape_functions_as_dict()

    def predict(self, X):
        return self.model.predict(X)



class IGANNAdapter(IGANN):
    def __init__(self, *args, **kwargs):
        super(IGANNAdapter, self).__init__(*args, **kwargs)


    def update_model_with_spline(self, selected_features, updated_data, X_train, y_train):
        # set weights of selected features to 0
        # add value from spline
        #print(selected_features)
        #print(updated_data)
        self.fit(X_train, y_train)
        x_axis = self.get_shape_functions_as_dict()
        print(type(x_axis))
        print(x_axis)
        print(x_axis[0])
        #print(X_train["age"])




    def replace_shape_curve(self, feature_name, new_x, new_y):
        # Implementation for replacing shape curve
        pass

    def nullify_regressor(self, feature_name):
        # Implementation to nullify a regressor
        pass

    def _add_function(self, function_name):
        # Implementation to add a new function
        pass
