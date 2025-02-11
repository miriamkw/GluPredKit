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
        return df, df

