"""
This is a preprocessor that is aiming to reproduce the steps described in the following paper:
https://pubmed.ncbi.nlm.nih.gov/32091990/

Note that, because of the inherent splitting of the Ohio data into a train and a test data folder,
the ohio parser and preprocessor splits based on that, and not from the test_data fraction in the configuration.
"""

from .base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np


class Preprocessor(BasePreprocessor):
    def __init__(self, numerical_features, categorical_features, prediction_horizon, num_lagged_features, test_size):
        super().__init__(numerical_features, categorical_features, prediction_horizon, num_lagged_features, test_size)

    def __call__(self, df):

        train_df, test_df = self.preprocess(df)
        return train_df, test_df

    def preprocess(self, df):
        is_test_col = 'is_test'
        train_df = df[df[is_test_col] == False]
        test_df = df[df[is_test_col] == True]

        # Drop columns that are not included
        train_df = train_df[self.numerical_features + self.categorical_features]
        test_df = test_df[self.numerical_features + self.categorical_features]

        # Check if any numerical features have NaN values before imputation, add a column "flag"
        test_df.loc[:, 'imputed'] = test_df.loc[:, self.numerical_features].isna().any(axis=1)

        # Add target for test data before interpolation to perceive NaN values
        test_df = self.add_target(test_df)

        # Interpolation using linear interpolation, with the maximum of 12 samples
        train_df = train_df.sort_index()
        train_df[self.numerical_features] = train_df[self.numerical_features].interpolate(method='linear', limit=12)
        test_df = test_df.sort_index()
        test_df[self.numerical_features] = test_df[self.numerical_features].interpolate(method='linear', limit=12)

        # Add target for train data after interpolation to use interpolated data for model training
        train_df = self.add_target(train_df)

        train_df = train_df.dropna()
        test_df = test_df.dropna(subset=self.numerical_features + self.categorical_features)

        # Scaling and transforming data
        train_df.loc[:, 'CGM'] = train_df.loc[:, 'CGM'] / 120
        test_df.loc[:, 'CGM'] = test_df.loc[:, 'CGM'] / 120

        train_df.loc[:, 'bolus'] = train_df.loc[:, 'bolus'] / 100
        test_df.loc[:, 'bolus'] = test_df.loc[:, 'bolus'] / 100

        train_df.loc[:, 'carbs'] = train_df.loc[:, 'carbs'] / 200
        test_df.loc[:, 'carbs'] = test_df.loc[:, 'carbs'] / 200

        train_df.loc[:, 'exercise'] = train_df['exercise'].apply(lambda x: 1 if x > 0 else 0)
        test_df.loc[:, 'exercise'] = test_df['exercise'].apply(lambda x: 1 if x > 0 else 0)

        return train_df, test_df


    def add_target(self, df):
        target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df = df.copy()
        df.loc[:, 'target'] = df['CGM'].shift(-target_index)

        # Check if 'imputed' column exists and handle it accordingly
        if 'imputed' in df.columns:
            # Create a shifted 'imputed' column
            shifted_imputed = df['imputed'].shift(-target_index)

            # Use logical OR to combine current and shifted 'imputed' status
            df['imputed'] = df['imputed'] | shifted_imputed

        return df
