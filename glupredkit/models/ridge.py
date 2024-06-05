from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from sklearn.multioutput import MultiOutputRegressor
import json
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.subject_ids = None
        self.models = []

    def _fit_model(self, x_train, y_train, *args):
        self.subject_ids = x_train['id'].unique()

        # Define the parameter grid
        param_grid = {
            'estimator__alpha': [0.001, 0.01, 0.1, 1.0]
        }
        for _ in range(len(self.subject_ids)):
            # Define the base regressor
            base_regressor = Ridge(tol=1)

            # Wrap the base regressor with MultiOutputRegressor
            multi_output_regressor = MultiOutputRegressor(base_regressor)

            # Perform grid search to find the best parameters and fit the model
            model = GridSearchCV(multi_output_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
            model.fit(x_train, y_train)

            self.models = self.models + [model]
        return self

    def _predict_model(self, x_test):
        y_pred = []
        ids_list = x_test.id.unique()

        for curr_id in ids_list:
            model_index = np.where(self.subject_ids == curr_id)[0][0]
            subset_df = x_test[x_test['id'] == curr_id]

            # Use the best estimator found by GridSearchCV to make predictions
            predictions = self.models[model_index].best_estimator_.predict(subset_df)
            y_pred += predictions.tolist()
        y_pred = np.array(y_pred)

        return y_pred

    def best_params(self):
        best_params = []

        for model in self.models:
            # Access the best estimator from GridSearchCV
            best_regressor = model.best_estimator_

            # Access the list of estimators (Ridge regressors) from MultiOutputRegressor
            ridge_regressors = best_regressor.estimators_

            # Iterate over the fitted Ridge regressors and add the alpha for each output
            for i, ridge_regressor in enumerate(ridge_regressors):
                best_params = best_params + [ridge_regressor.alpha]

        # Return the best parameters found by GridSearchCV
        return best_params

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def print_coefficients(self):
        for model in self.models:
            # Access the best estimator from GridSearchCV
            best_regressor = model.best_estimator_

            # Access the list of estimators (Ridge regressors) from MultiOutputRegressor
            ridge_regressors = best_regressor.estimators_

            # Iterate over the fitted Ridge regressors and add the alpha for each output
            for i, ridge_regressor in enumerate(ridge_regressors):
                print(f'Coefficients for model {i}')
                feature_names = ridge_regressor.feature_names_in_
                coefficients = ridge_regressor.coef_
                for feature_name, coefficient in zip(feature_names, coefficients):
                    print(f"Feature: {feature_name}, Coefficient: {coefficient:.4f}")

    # This method saves the model weights to a json file. It is useful if you want to use the model in for example
    # a real-time smartphone application
    def save_model_weights(self, file_path):
        # Extract coefficients and intercepts for each output
        coefficients = [estimator.coef_ for estimator in self.model.best_estimator_.estimators_]
        intercepts = [estimator.intercept_ for estimator in self.model.best_estimator_.estimators_]

        # Convert coefficients to a list of lists
        coefficients_list = [coef.tolist() for coef in coefficients]

        # Convert numpy arrays to lists to ensure JSON serialization
        coefficients_list = [[float(value) for value in coef_row] for coef_row in coefficients_list]
        intercepts_list = [float(value) for value in intercepts]

        # Convert feature names to a list
        feature_names = self.model.best_estimator_.feature_names_in_
        feature_names_list = feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names

        # Create a dictionary to store the model weights
        model_weights = {
            "n_outputs": len(coefficients),
            "n_features": len(coefficients[0]),
            "feature_names": feature_names_list,
            "coefficients": coefficients_list,
            "intercepts": intercepts_list,
        }

        # Save the model weights to a JSON file
        with open(file_path, "w") as f:
            json.dump(model_weights, f, indent=4)  # Use indent for pretty printing

