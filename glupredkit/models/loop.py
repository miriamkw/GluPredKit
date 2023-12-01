from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
# Run pip install git+https://github.com/miriamkw/pyloopkit.git@parabolic_meal_model for this dependency
from pyloopkit.exponential_insulin_model import percent_effect_remaining


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

    def fit(self, x_train, y_train):
        # TODO: Fit an average basal rate?

        return self

    def predict(self, x_test):
        # Use the best estimator found by GridSearchCV to make predictions
        y_pred = x_test["CGM"]

        # TODO: For each insulin dose, add the glucose effects

        return y_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)



def insulin_dose_glucose_effect():
    """
    This function calculates how much one dose of insulin is estimated to lower blood glucose in the interval between
    the calculation date and the predicted value.

    Inputs:
    - Basal rate
    - Insulin sensitivity
    - Value of insulin dose
    - Time of insulin dose
    - Time of blood glucose measurement
    - Time of blood glucose prediction

    Returns:
    The glucose effect from this insulin dose between measured and predicted value
    """
    # TODO: Calculate percentage of effect remaining for measurement
    # TODO: Calculate percentage of effect remaining for prediction









