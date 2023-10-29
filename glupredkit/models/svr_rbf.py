from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.model = None

    def fit(self, x_train, y_train):

        # Define the model
        pipeline = Pipeline([ ('regressor', SVR(tol=1, kernel='linear'))])

        # Define the parameter grid
        param_grid = {
            'regressor__C': [100],
            'regressor__epsilon': [0.03],
            'regressor__gamma': [0.008],
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
