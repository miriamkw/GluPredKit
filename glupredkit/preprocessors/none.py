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

        dataset_ids = df['id'].unique()
        dataset_ids = list(filter(lambda x: not np.isnan(x), dataset_ids))

        processed_train_df = pd.DataFrame()
        processed_test_df = pd.DataFrame()

        for subject_id in dataset_ids:
            subset_df_train = train_df[train_df['id'] == subject_id]
            subset_df_test = test_df[test_df['id'] == subject_id]

            subset_df_train_with_targets = self.add_targets(subset_df_train)
            subset_df_test_with_targets = self.add_targets(subset_df_test)

            processed_test_df = pd.concat([processed_test_df, subset_df_train_with_targets], axis=0)
            processed_test_df = pd.concat([processed_test_df, subset_df_test_with_targets], axis=0)

        return processed_train_df, processed_test_df

    def add_targets(self, df):
        max_target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df = df.copy()

        for i in range(1, max_target_index + 1):
            df.loc[:, f'target_{i * 5}'] = df[f'CGM'].shift(-i)

        return df


