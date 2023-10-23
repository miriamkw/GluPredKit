from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.model = None

    def fit(self, x_train, y_train):

        # Define the model
        pipeline = Pipeline([('regressor', RandomForestRegressor())])

        # Define the parameter grid
        param_grid = {
            'regressor__n_estimators': [50, 100, 300, 500],
            'regressor__min_samples_split': [2, 20, 40, 80],
            'regressor__max_depth': [None, 10, 30, 50],
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

    def process_data(self, df, num_lagged_features, numerical_features, categorical_features):
        return process_data(df, num_lagged_features, numerical_features, categorical_features)
