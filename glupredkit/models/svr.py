from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.model = None

    def _fit_model(self, x_train, y_train, *args):
        # Define the base regressor
        base_regressor = SVR(tol=1, kernel='rbf')

        # Wrap the base regressor with MultiOutputRegressor
        multi_output_regressor = MultiOutputRegressor(base_regressor)

        # Define the parameter grid
        param_grid = {
            'estimator__C': [100],
            'estimator__epsilon': [0.03],
            'estimator__gamma': [0.008],
        }

        # Define GridSearchCV
        self.model = GridSearchCV(multi_output_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
        self.model.fit(x_train, y_train)
        return self

    def _predict_model(self, x_test):
        # Use the best estimator found by GridSearchCV to make predictions
        y_pred = self.model.best_estimator_.predict(x_test)
        return y_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return self.model.best_params_

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
