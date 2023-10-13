from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from .base_model import BaseModel


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.model = None

    def fit(self, x_train, y_train):
        # Define the model
        pipeline = Pipeline([('regressor', HuberRegressor(max_iter=1000, tol=1))])

        # Define the parameter grid
        param_grid = {
            'regressor__epsilon': [1.3, 1.35, 1.5, 1.75],
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