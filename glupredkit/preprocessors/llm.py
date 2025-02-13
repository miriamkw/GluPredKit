from .base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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
        train_df = df[~df['is_test']]
        test_df = df[df['is_test']]

        dataset_ids = train_df['id'].unique()
        dataset_ids = list(filter(lambda x: not np.isnan(x), dataset_ids))

        processed_train_df = pd.DataFrame()
        processed_test_df = pd.DataFrame()

        for subject_id in dataset_ids:
            subset_df_train = train_df[train_df['id'] == subject_id]
            subset_df_test = test_df[test_df['id'] == subject_id]

            expanded_subset_df_train = self.get_df_with_targets_and_lagged_features(subset_df_train)
            expanded_subset_df_test = self.get_df_with_targets_and_lagged_features(subset_df_test)

            # Add the processed data to the dataframes
            processed_train_df = pd.concat([processed_train_df, expanded_subset_df_train], axis=0)
            processed_test_df = pd.concat([processed_test_df, expanded_subset_df_test], axis=0)

        return processed_train_df, processed_test_df

    def get_df_with_targets_and_lagged_features(self, df):
        lagged_features = self.add_time_lagged_features(df, self.numerical_features,
                                                        self.num_lagged_features)
        what_if_df = self.add_what_if_features(df, self.what_if_features, self.prediction_horizon)
        df_with_targets = self.add_targets(df)

        # Update the subset DataFrame with the new time-lagged features
        expanded_df = pd.concat([df_with_targets, lagged_features], axis=1)
        expanded_df = pd.concat([expanded_df, what_if_df], axis=1)
        return expanded_df

    def add_targets(self, df):
        max_target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df = df.copy()

        for i in range(1, max_target_index + 1):
            df.loc[:, f'target_{i * 5}'] = df[f'CGM'].shift(-i)

        return df

    def add_time_lagged_features(self, df, lagged_cols, num_lagged_features):
        lagged_df = pd.DataFrame()
        indexes = list(range(1, num_lagged_features + 1))

        for col in lagged_cols:
            for i in indexes:
                new_col_name = col + "_" + str(i * 5)
                lagged_df = pd.concat([lagged_df, df[col].shift(i).rename(new_col_name)], axis=1)
        return lagged_df

    def add_what_if_features(self, df, what_if_cols, prediction_horizon):
        what_if_df = pd.DataFrame()
        prediction_index = prediction_horizon // 5
        indexes = list(range(1, prediction_index + 1))

        for col in what_if_cols:
            for i in indexes:
                new_col_name = col + "_what_if_" + str(i * 5)
                what_if_df = pd.concat([what_if_df, df[col].shift(-i).rename(new_col_name)], axis=1)
        return what_if_df



