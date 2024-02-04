# ENSEMBLE 2: Huber Regression + Gradient Boosting

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.ph = prediction_horizon
        self.model = None

    def fit(self, x_train, y_train):
        print("x_train", x_train)

        # define the models
        huber = HuberRegressor()
        gbr = GradientBoostingRegressor()

        # combine the models into an ensemble
        self.model = VotingRegressor([('huber', huber), ('gbr', gbr)])

        # fit the model
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        # make predictions
        return self.model.predict(x_test)
    

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)