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
        test_df = self.add_targets(test_df)

        # Interpolation using a nonlinear curve, without too much curvature
        train_df = train_df.sort_index()
        train_df[self.numerical_features] = train_df[self.numerical_features].interpolate(method='akima')
        test_df = test_df.sort_index()
        test_df[self.numerical_features] = test_df[self.numerical_features].interpolate(method='akima')

        # Add target for train data after interpolation to use interpolated data for model training
        train_df = self.add_targets(train_df)

        train_df = train_df.dropna()
        test_df = test_df.dropna(subset=self.numerical_features + self.categorical_features)

        # Transform columns
        if self.numerical_features:
            """
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Fit the scaler only on training data
            scaler.fit(train_df.loc[:, self.numerical_features])

            # Transform data
            train_df.loc[:, self.numerical_features] = scaler.transform(train_df.loc[:, self.numerical_features])
            test_df.loc[:, self.numerical_features] = scaler.transform(test_df.loc[:, self.numerical_features])
            """
        if self.categorical_features:
            encoder = OneHotEncoder(drop='first')  # dropping the first column to avoid dummy variable trap

            # Fit the encoder only on training data
            encoder.fit(train_df[self.categorical_features])

            # Transform data
            train_df = self.transform_with_encoder(train_df, encoder)
            test_df = self.transform_with_encoder(test_df, encoder)

        return train_df, test_df

    def transform_with_encoder(self, df, encoder):
        encoded_cols = encoder.transform(df.loc[:, self.categorical_features])
        encoded_df = pd.DataFrame(encoded_cols.toarray(),
                                  columns=encoder.get_feature_names_out(self.categorical_features),
                                  index=df.index)
        df = df.drop(columns=self.categorical_features)
        df = pd.concat([df, encoded_df], axis=1)
        return df

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

    def add_targets(self, df):
        max_target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df = df.copy()

        for i in range(1, max_target_index + 1):
            df.loc[:, f'target_{i * 5}'] = df[f'CGM'].shift(-i)

        # Check if 'imputed' column exists and handle it accordingly
        if 'imputed' in df.columns:
            # Create a shifted 'imputed' column
            shifted_imputed = df['imputed'].shift(-max_target_index)

            # Use logical OR to combine current and shifted 'imputed' status
            df['imputed'] = df['imputed'] | shifted_imputed

        return df
