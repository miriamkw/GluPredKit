"""
This is a preprocessor that is aiming to reproduce the steps described in the following paper:
https://pubmed.ncbi.nlm.nih.gov/32091990/

NOTE: THIS PREPROCESSOR IS NOT WORKING AS INTENDED! Hence, it is not added to the CLI.
"""

from .base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, Normalizer


# TODO: ADD, IF NOT TRAINING DATA, DONT INCLUDE SAMPLES ADDED BY IMPUTATION
class Preprocessor(BasePreprocessor):
    def __init__(self, numerical_features, categorical_features, prediction_horizon, num_lagged_features, test_size):
        super().__init__(numerical_features, categorical_features, prediction_horizon, num_lagged_features, test_size)

    def __call__(self, df):

        # Drop columns that are not included
        df = df[self.numerical_features + self.categorical_features]

        # Check if any numerical features have NaN values before imputation
        df['imputed'] = df[self.numerical_features].isna().any(axis=1)

        # Interpolation using forward fill
        df[self.numerical_features] = df[self.numerical_features].interpolate(method='polynomial', order=2)

        # Add target column
        target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df = df.copy()
        df.loc[:, 'target'] = df['CGM'].shift(-target_index)

        df = df.dropna()

        # Transform columns
        if self.numerical_features:
            scaler = Normalizer()
            df.loc[:, self.numerical_features] = scaler.fit_transform(df.loc[:, self.numerical_features])
        if self.categorical_features:
            encoder = OneHotEncoder(drop='first')  # dropping the first column to avoid dummy variable trap
            encoded_cols = encoder.fit_transform(df[self.categorical_features])
            encoded_df = pd.DataFrame(encoded_cols.toarray(),
                                      columns=encoder.get_feature_names_out(self.categorical_features), index=df.index)
            df = df.drop(columns=self.categorical_features)
            df = pd.concat([df, encoded_df], axis=1)

        # Since the Ohio T1DM dataset is split into train and test data in the folder structure, users must
        # update their configurations to use the correct data instead of splitting inherently in the preprocessor
        return df, df

