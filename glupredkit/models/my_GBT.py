import xgboost as xgb
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.model = None

    def fit(self, x_train, y_train):
        # Define the model
        pipeline = Pipeline([('regressor', xgb.XGBRegressor())])

        # Define the parameter grid
        ''' 
        # Got better results with the following parameters but with significantly longer training time
        param_grid = {
            'regressor__n_estimators': [1000, 2000, 3000],
            'regressor__max_depth': [8, 10, 12],
            'regressor__gamma': [0.1, 0.2, 0.3],
            'regressor__learning_rate': [0.01, 0.05, 0.1]
        }
        '''
        # Tuned the parameters for better performance (Ohio dateset)
        param_grid = {
            'regressor__n_estimators': [1000],
            'regressor__max_depth': [8],
            'regressor__gamma': [0.1],
            'regressor__learning_rate': [0.01]
        }

        # Define GridSearchCV
        self.model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        self.model.fit(x_train, y_train, regressor__early_stopping_rounds=10, regressor__eval_set=[(x_train, y_train)])
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