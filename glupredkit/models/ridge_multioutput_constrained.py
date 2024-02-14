from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
import json
import numpy as np
import cvxpy as cp
import pandas as pd

class Model(BaseModel):
    def __init__(self, prediction_horizon, lambda_ridge=0.1):
        super().__init__(prediction_horizon)

        self.model = None
        self.lambda_ridge = lambda_ridge  # Regularization parameter for Ridge regression
        self.beta = None  # To store the model coefficients
        self.intercept = None  # To store the model intercepts
        self.feature_names = []  # To store feature names for identifying insulin columns

    def fit(self, x_train, y_train):
        self.feature_names = x_train.columns.tolist()
        n_features = x_train.shape[1]
        n_outputs = y_train.shape[1]

        # Initialize storage for model coefficients and intercepts for each output
        self.beta = np.zeros((n_features, n_outputs))
        self.intercept = np.zeros(n_outputs)

        for output_index in range(n_outputs):
            # Define the optimization variables
            beta = cp.Variable(n_features)
            intercept = cp.Variable(1)

            # Target column
            y_target = y_train.iloc[:, output_index].values

            # Define the loss function (Ridge)
            loss = cp.sum_squares(
            cp.matmul(x_train.values, beta) + intercept - y_target) + self.lambda_ridge * cp.sum_squares(beta)

            # Constraints: negative coefficients for features starting with "insulin", positive for carbs
            constraints = [beta[i] <= 0 for i, name in enumerate(self.feature_names) if name.startswith('insulin')]
            constraints = constraints + [beta[i] >= 0 for i, name in enumerate(self.feature_names) if
                                         name.startswith('carbs')]

            # TODO: Add more constraints if physiologically reasonable

            # Define and solve the problem
            problem = cp.Problem(cp.Minimize(loss), constraints)
            problem.solve()

            # Store the coefficients and intercept for this output
            self.beta[:, output_index] = beta.value
            self.intercept[output_index] = intercept.value

        return self

    def predict(self, x_test):
        if self.beta is None or self.intercept is None:
            raise Exception("Model has not been fitted.")

        # Number of outputs
        n_outputs = self.beta.shape[1]

        # Initialize an array to store predictions for each output
        y_pred = np.zeros((x_test.shape[0], n_outputs))

        # Calculate predictions for each output
        for output_index in range(n_outputs):
            # Ensure x_test is a numpy array for matrix operations
            y_pred[:, output_index] = np.dot(x_test.values, self.beta[:, output_index]) + self.intercept[output_index]

        return y_pred

    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def print_coefficients(self):
        # Number of outputs
        n_outputs = self.beta.shape[1]

        # Iterate over the fitted Ridge regressors and add the alpha for each output
        for output_index in range(n_outputs):
            print(f'Coefficients for model {output_index}')
            coefficients = self.beta[:, output_index]
            for feature_name, coefficient in zip(self.feature_names, coefficients):
                print(f"Feature: {feature_name}, Coefficient: {coefficient:.4f}")

    def save_model_weights(self, file_path):
        # Convert coefficients to a list of lists
        coefficients_list = self.beta.T.tolist()

        # Create a dictionary to store the model weights
        model_weights = {
            "n_outputs": self.beta.shape[1],
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "coefficients": coefficients_list,
            "intercepts": self.intercept.tolist(),
        }

        # Save the model weights to a JSON file
        with open(file_path, "w") as f:
            json.dump(model_weights, f, indent=4)  # Use indent for pretty printing


