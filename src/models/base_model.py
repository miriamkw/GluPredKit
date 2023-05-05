import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df_glucose, df_bolus, df_basal, df_carbs):
        # Perform any additional processing of the input features here

        # Fit the model
        # ...

        raise NotImplementedError("Model has not implemented fit method!")

    def predict(self, df_glucose, df_bolus, df_basal, df_carbs):
        # Perform any additional processing of the input features here

        # Make predictions using the fitted model
        # ...

        # Return the predictions
        raise NotImplementedError("Model has not implemented predict method!")

    def predict_one_prediction(self, df_glucose, df_bolus, df_basal, df_carbs):
        """
        This function predicts one prediction based on the most recent data in the input.

        Returns a list of values containing the predicted trajectory for the model.
        """
        # Perform any additional processing of the input features here

        # Make predictions using the fitted model
        # ...

        # Return the predictions
        raise NotImplementedError("Model has not implemented predict_one_prediction method!")
