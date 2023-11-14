from skimage.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from igann import IGANN
import matplotlib.pyplot as plt
import numpy as np

class Adapter():
    def __init__(self, model):
        self.m = model

    def replace_shape_curve(self, feature_name, new_x, new_y, method="curve_fitting"):
        if method == "curve_fitting":
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





if __name__ == '__main__':
    # Generate 250 x points from 0 onwards
    x_points = np.linspace(0, 10, 250)


    # Adjust the custom function to apply only for x >= 0
    def custom_function(x):
        if 2 < x < 4:  # Let's change the constant range for the sake of variety
            return 5  # Constant part
        else:
            return 0.1 * x ** 2  # Non-linear part


    # Generate y points based on the custom function for x >= 0
    y_points = np.array([custom_function(x) for x in x_points])

    # Plotting the graph
    plt.figure(figsize=(12, 6))
    plt.plot(x_points, y_points, label='Non-linear function with constant part')
    plt.title('Non-linear Graph with Constant Range (x >= 0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    from scipy.interpolate import CubicSpline, CubicHermiteSpline

    # Create a cubic spline interpolation of the generated points
    cs = CubicSpline(x_points, y_points)

    # Generate a dense range of x values for plotting the smooth spline curve
    x_dense = np.linspace(0, 10, 100)
    y_dense = cs(x_dense)

    # Plotting the original points and the spline interpolation
    plt.figure(figsize=(12, 6))
    plt.plot(x_points, y_points, 'o', label='Original points')
    plt.plot(x_dense, y_dense, label='Cubic spline interpolation')
    plt.title('Cubic Spline Interpolation of Non-linear Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    #get_shape_functions_as_dict