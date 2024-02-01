from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data

# Gradient Boosting + Random Forest

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.model = None

    def fit(self, x_train, y_train):
        # define the models
        gbt = xgb.XGBRegressor(n_estimators=1000, max_depth=8, gamma=0.1, learning_rate=0.01)
        rf = RandomForestRegressor(n_estimators=1000, max_depth=8, random_state=42)

        # combine the predictions of a Gradient Boosting model and a Random Forest model.
        self.model = VotingRegressor([('gbt', gbt), ('rf', rf)])

        # Fit the model
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        # Use the ensemble model to make predictions
        y_pred = self.model.predict(x_test)
        return y_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)