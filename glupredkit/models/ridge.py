from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.model = None

    def fit(self, x_train, y_train):
        # Perform grid search to find the best parameters and fit the model

        # Define the model
        pipeline = Pipeline([('regressor', Ridge(tol=1))])

        # Define the parameter grid
        param_grid = {
            'regressor__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        # Define GridSearchCV
        self.model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        # Use the best estimator found by GridSearchCV to make predictions
        y_pred = self.model.best_estimator_.predict(x_test)
        return y_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return self.model.best_params_

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def print_coefficients(self):
        feature_names = self.model.best_estimator_[0].feature_names_in_
        coefficients = self.model.best_estimator_[0].coef_
        for feature_name, coefficient in zip(feature_names, coefficients):
            print(f"Feature: {feature_name}, Coefficient: {coefficient:.4f}")
