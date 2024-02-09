"""
The basic preprocessor does the following:
- Standardize numerical features
- One hot encode categorical features
- Add the target value
"""
from .base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor(BasePreprocessor):
    def __init__(self, numerical_features, categorical_features, what_if_features, prediction_horizon,
                 num_lagged_features, test_size):
        super().__init__(numerical_features, categorical_features, what_if_features, prediction_horizon,
                         num_lagged_features, test_size)

    def __call__(self, df):

        # Drop columns that are not included
        df = df[self.numerical_features + self.categorical_features]

        # Add target column
        target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df = df.copy()

        # Check if the CGM value has NaN values before imputation, add a column "flag"
        df.loc[:, 'imputed'] = df.loc[:, ["CGM"]].isna().any(axis=1)
        df = self.add_targets(df)

        # Interpolation using forward fill
        df[self.numerical_features] = df[self.numerical_features].interpolate(limit_direction='both')

        # Train and test split
        # Adding a margin of 24 hours to the train and the test data to avoid memory leak
        margin = int((12 * 24 + self.num_lagged_features + target_index) / 2)
        split_index = int((len(df) - margin) * (1 - self.test_size))

        # Split the data into train and test sets
        train_data = df[:split_index - margin]
        test_data = df[split_index + margin:]

        if self.categorical_features:
            encoder = OneHotEncoder(drop='first')  # dropping the first column to avoid dummy variable trap

            # Fit the encoder only on training data
            encoder.fit(train_data[self.categorical_features])

            # Transform data
            train_data = self.transform_with_encoder(train_data, encoder)
            test_data = self.transform_with_encoder(test_data, encoder)

        train_data = self.add_what_if(train_data)
        test_data = self.add_what_if(test_data)

        train_data = train_data.drop(columns=['imputed'])

        return train_data, test_data

    def transform_with_encoder(self, df, encoder):
        encoded_cols = encoder.transform(df.loc[:, self.categorical_features])
        encoded_df = pd.DataFrame(encoded_cols.toarray(),
                                  columns=encoder.get_feature_names_out(self.categorical_features),
                                  index=df.index)
        df = df.drop(columns=self.categorical_features)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    # The prediction horizon is the longest prediction horizon.
    # We assume targets for 5-minute intervals
    # TODO: Fix so that imputed shows true for any part of the trajectory of imputed values
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

    def add_what_if(self, df):
        max_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")

        df = df.copy()

        for col_name in self.what_if_features:
            for i in range(1, max_index + 1):
                df.loc[:, f'{col_name}_what_if_{i * 5}'] = df[col_name].shift(-i)

        return df

