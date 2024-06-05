from .base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


class Preprocessor(BasePreprocessor):
    def __init__(self, subject_ids, numerical_features, categorical_features, what_if_features, prediction_horizon,
                 num_lagged_features):
        super().__init__(subject_ids, numerical_features, categorical_features, what_if_features, prediction_horizon,
                         num_lagged_features)

    def __call__(self, df):
        train_df, test_df = self.preprocess(df)
        return train_df, test_df

    def preprocess(self, df):
        # Filter out subject ids if not configuration is None
        if self.subject_ids:
            df = df[df['id'].isin(self.subject_ids)]

        train_df = df[~df['is_test']]
        test_df = df[df['is_test']]

        dataset_ids = train_df['id'].unique()
        dataset_ids = list(filter(lambda x: not np.isnan(x), dataset_ids))

        # Drop columns that are not included
        train_df = train_df[self.numerical_features + self.categorical_features + ['id']]
        test_df = test_df[self.numerical_features + self.categorical_features + ['id']]

        # Check if any numerical features have NaN values before imputation, add a column "flag"
        test_df.loc[:, 'imputed'] = test_df.loc[:, self.numerical_features].isna().any(axis=1)

        processed_train_df = pd.DataFrame()
        processed_test_df = pd.DataFrame()

        for subject_id in dataset_ids:
            subset_df_train = train_df[train_df['id'] == subject_id]
            subset_df_test = test_df[test_df['id'] == subject_id]

            # Interpolation using a nonlinear curve, without too much curvature
            subset_df_train = subset_df_train.sort_index()
            subset_df_train[self.numerical_features] = (subset_df_train[self.numerical_features]
                                                        .interpolate(method='akima'))

            # Add test columns before interpolation to perceive nan values
            subset_test_df_with_targets = self.add_targets(subset_df_test)

            subset_test_df_with_targets = subset_test_df_with_targets.sort_index()
            subset_test_df_with_targets[self.numerical_features] = (subset_test_df_with_targets[self.numerical_features]
                                                                    .interpolate(method='akima'))

            # Add target for train data after interpolation to use interpolated data for model training
            subset_train_df_with_targets = self.add_targets(subset_df_train)

            # Transform columns
            if self.numerical_features:
                scaler = StandardScaler()

                # Fit the scaler only on training data
                scaler.fit(subset_train_df_with_targets.loc[:, self.numerical_features])

                # Transform data
                subset_train_df_with_targets.loc[:, self.numerical_features] = (
                    scaler.transform(subset_train_df_with_targets.loc[:, self.numerical_features]))
                subset_test_df_with_targets.loc[:, self.numerical_features] = (
                    scaler.transform(subset_test_df_with_targets.loc[:, self.numerical_features]))

            # Add the processed data to the dataframes
            processed_train_df = pd.concat([processed_train_df, subset_train_df_with_targets], axis=0)
            processed_test_df = pd.concat([processed_test_df, subset_test_df_with_targets], axis=0)

        if self.categorical_features:
            encoder = OneHotEncoder(drop='first')  # dropping the first column to avoid dummy variable trap

            # Fit the encoder only on training data
            encoder.fit(train_df[self.categorical_features])

            # Transform data
            processed_train_df = self.transform_with_encoder(processed_train_df, encoder)
            processed_test_df = self.transform_with_encoder(processed_test_df, encoder)

        return processed_train_df, processed_test_df

    def transform_with_encoder(self, df, encoder):
        encoded_cols = encoder.transform(df.loc[:, self.categorical_features])
        encoded_df = pd.DataFrame(encoded_cols.toarray(),
                                  columns=encoder.get_feature_names_out(self.categorical_features),
                                  index=df.index)
        df = df.drop(columns=self.categorical_features)
        df = pd.concat([df, encoded_df], axis=1)
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
