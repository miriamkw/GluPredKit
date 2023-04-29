import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BaseModel(BaseEstimator, TransformerMixin):
    def __init__(self, output_offsets=None):
        if output_offsets is None:
            self.output_offsets = list(range(5, 361, 5)) # default list from 5 to 360 with step 5 like in Loop
        else:
            self.output_offsets = output_offsets

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

    def get_output_offsets(self):
        return self.output_offsets