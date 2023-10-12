"""
The scikit learn preprocessor takes in a parsed dataset in a 5-minute time grid and returns a dataset ready to be
trained with a model that builds on the scikit learn library.

Users can customize:
- History length
- Prediction horizon for target
- Test size
- Whether to include hour of day
- A list of categorical features
- A list of numerical features
"""
from .base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_time_lagged_features(col_name, df, num_lagged_features):
    indexes = list(range(1, num_lagged_features + 1))

    for i in indexes:
        new_col_name = col_name + "_" + str(i * 5)
        df = pd.concat([df, df[col_name].shift(i).rename(new_col_name)], axis=1)
    return df


class Preprocessor(BasePreprocessor):
    def __init__(self, numerical_features, categorical_features, prediction_horizon, num_lagged_features, test_size,
                 include_hour):
        super().__init__(numerical_features, categorical_features, prediction_horizon, num_lagged_features, test_size,
                         include_hour)

    def __call__(self, df):

        # Imputation of CGM values if there is a one-sample gap
        df['CGM'] = df.CGM.ffill(limit=1)

        # Add hour of day
        if self.include_hour:
            df['hour'] = df.index.copy().to_series().apply(lambda x: x.hour)

        # Drop columns that are not included
        df = df[self.numerical_features + self.categorical_features]

        # Add target column
        target_index = self.prediction_horizon // 5
        if self.prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df['target'] = df.CGM.shift(-target_index)

        # Transform columns
        if self.numerical_features:
            df[self.numerical_features] = StandardScaler().fit_transform(df[self.numerical_features])
        if self.categorical_features:
            encoder = OneHotEncoder(drop='first')  # dropping the first column to avoid dummy variable trap
            encoded_cols = encoder.fit_transform(df[self.categorical_features])
            encoded_df = pd.DataFrame(encoded_cols.toarray(),
                                      columns=encoder.get_feature_names_out(self.categorical_features), index=df.index)
            df = df.drop(columns=self.categorical_features)
            df = pd.concat([df, encoded_df], axis=1)

        # Add time-lagged features
        for col in self.numerical_features:
            df = add_time_lagged_features(col, df, self.num_lagged_features)

        df = df.dropna()

        # Train and test split
        # Adding a margin of 24 hours to the train and the test data to avoid memory leak
        margin = int((12 * 24 + self.num_lagged_features + target_index) / 2)
        split_index = int((len(df) - margin) * (1 - self.test_size))

        # Split the data into train and test sets
        train_data = df[:split_index - margin]
        test_data = df[split_index + margin:]

        return train_data, test_data
