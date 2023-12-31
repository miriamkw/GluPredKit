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

    def __call__(self, df, **kwargs):

        # Drop columns that are not included
        df = df[self.numerical_features + self.categorical_features]

        # Add target column
        target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df = df.copy()

        # Check if the CGM value has NaN values before imputation, add a column "flag"
        df.loc[:, 'imputed'] = df.loc[:, ["CGM"]].isna().any(axis=1)
        df = self.add_target(df)

        # Interpolation using forward fill
        df[self.numerical_features] = df[self.numerical_features].ffill()

        # Train and test split
        # Adding a margin of 24 hours to the train and the test data to avoid memory leak
        margin = int((12 * 24 + self.num_lagged_features + target_index) / 2)
        split_index = int((len(df) - margin) * (1 - self.test_size))

        # Split the data into train and test sets
        train_data = df[:split_index - margin]
        test_data = df[split_index + margin:]

        # Transform columns
        if self.numerical_features:
            scaler = StandardScaler()

            # Fit the scaler only on training data
            scaler.fit(train_data.loc[:, self.numerical_features])

            # Transform data
            train_data.loc[:, self.numerical_features] = scaler.transform(train_data.loc[:, self.numerical_features])
            test_data.loc[:, self.numerical_features] = scaler.transform(test_data.loc[:, self.numerical_features])

        if self.categorical_features:
            encoder = OneHotEncoder(drop='first')  # dropping the first column to avoid dummy variable trap

            # Fit the encoder only on training data
            encoder.fit(train_data[self.categorical_features])

            # Transform data
            train_data = self.transform_with_encoder(train_data, encoder)
            test_data = self.transform_with_encoder(test_data, encoder)

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
