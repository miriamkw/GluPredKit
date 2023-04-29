"""
This linear regressor is provided as a simple example of how this framework can be used.
"""
from sklearn.linear_model import LinearRegression
from src.models.base_model import BaseModel
from typing import List

class LinearRegressor(BaseModel):
    def __init__(self, output_offsets: List[int] = None):
        # TODO: Implement support for several prediction horizons
        if output_offsets is None:
            output_offsets = list(range(5, 365, 5)) # default offsets
        self.output_offsets = output_offsets
        self.model = LinearRegression()

    def fit(self, df_glucose, df_bolus, df_basal, df_carbs):
        # concatenate dataframes and target into X and y
        X, y = self.process_data(df_glucose)

        # fit the model
        self.model.fit(X, y)
        return self

    def predict(self, df_glucose, df_bolus, df_basal, df_carbs):
        X_test, y_test = self.process_data(df_glucose)
        y_pred = self.model.predict(X_test)

        return y_pred, y_test

    def process_data(self, df_glucose):
        # Assuming only one output, finding the index of the offset
        index_offset = int(self.output_offsets[0]/5)
        target_column_name = str(self.output_offsets[0])

        data = df_glucose.copy()
        data[target_column_name] = data['value'].shift(index_offset)
        data = data.dropna()

        X = data[['value']]
        y = data[target_column_name]

        return X, y

