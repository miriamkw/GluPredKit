"""
The scikit learn preprocessor takes in a parsed dataset in a 5-minute time grid and returns a dataset ready to be
trained with a model that builds on the scikit learn library.

Users can customize:
- History length
- Prediction horizon for target
- Test size
- Whether to include hour of day
TODO: Add categorical features vs numerical features?
"""
from .base_preprocessor import BasePreprocessor
import pandas as pd


def add_time_lagged_features(col_name, df, num_lagged_features):
    indexes = list(range(1, num_lagged_features + 1))

    for i in indexes:
        new_col_name = col_name + "_" + str(i * 5)
        df = pd.concat([df, df[col_name].shift(i).rename(new_col_name)], axis=1)
    return df


class Preprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__

    def __call__(self, df, prediction_horizon=60, num_lagged_features=12, include_hour=True, test_size=0.2):
        """
        Generate time-lagged features for a dataset.

        Args:
            data (DataFrame): The input dataset.
            prediction_horizon (int): The prediction horizon in minutes.
            num_lagged_features (int): The number of time-lagged features to generate (12 samples corresponds to one
            hour).
            include_hour (bool): Whether to include the hour of the day as a feature.
            test_size (float): The fraction of data to reserve for testing.

        Returns:
            train_data: The dataset for model training.
            test_data: The dataset for model testing.
        """

        # Imputation of CGM values if there is a one-sample gap
        df['CGM'] = df.CGM.ffill(limit=1)

        # Add hour of day
        if include_hour:
            df['hour'] = df.index.copy().to_series().apply(lambda x: x.hour)

        # Add time-lagged features
        numerical_features = ['CGM', 'insulin', 'carbs']
        for col in numerical_features:
            df = add_time_lagged_features(col, df, num_lagged_features)

        # Add target column
        target_index = prediction_horizon // 5
        if prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df['target'] = df.CGM.shift(-target_index)

        # Train and test split
        split_index = int(len(df) * (1 - test_size))
        # Adding a margin of 24 hours to the train and the test data to avoid memory leak
        margin = 12 * 24 + num_lagged_features + target_index

        # Split the data into train and test sets
        train_data = df[:split_index - margin]
        test_data = df[split_index + margin:]

        return train_data, test_data

