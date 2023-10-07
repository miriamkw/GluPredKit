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
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Storing the data into a 3d-format
def create_dataframe(sequences, targets):
    # Create a new DataFrame to store sequences and targets

    # Convert sequences to lists
    # sequences_as_lists = [list(seq) for seq in sequences]
    # sequences_as_strings = [str(seq) for seq in sequences]
    sequences_as_strings = [str(list(map(list, seq))) for seq in sequences]


    dataset_df = pd.DataFrame({
        'sequence': sequences_as_strings, # sequences_as_lists,
        'target': targets
    })

    return dataset_df


def _prepare_sequences(data, labels, window_size, step_size=1):
    X, y = [], []

    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:i + window_size])
        y.append(labels[i + window_size])

    return np.array(X), np.array(y)


class Preprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()
        self.scalers = {}
        # TODO: Add this dynamically
        self.categorical_features = []
        self.numerical_features = ['CGM', 'insulin', 'carbs']

    def __call__(self, df, prediction_horizon, num_lagged_features=12, include_hour=True, test_size=0.2):
        """
        Process raw data from data/raw to data ready for training and testing. The Tensorflow-preprocessor is
        designed to process data for models trained using the Tensorflow library, usually neural networks like
        LSTM and TCN.
        """

        # Imputation of CGM values if there is a one-sample gap
        df['CGM'] = df.CGM.ffill(limit=1)

        # Add hour of day
        if include_hour:
            df['hour'] = df.index.copy().to_series().apply(lambda x: x.hour)

        # Add target column
        target_index = prediction_horizon // 5
        if prediction_horizon % 5 != 0:
            raise ValueError("Prediction horizon must be divisible by 5.")
        df['target'] = df.CGM.shift(-target_index)

        df = df.dropna()

        df_X, df_y = df.drop("target", axis=1), df["target"]
        processed_data = self._preprocess_data(df_X)

        sequences, targets = _prepare_sequences(processed_data, df_y, window_size=num_lagged_features)
        df = create_dataframe(sequences, targets)

        # Train and test split
        # Adding a margin of 24 hours to the train and the test data to avoid memory leak
        margin = int((12 * 24 + num_lagged_features + target_index) / 2)
        split_index = int((len(df) - margin) * (1 - test_size))

        # Split the data into train and test sets
        train_data = df[:split_index - margin]
        test_data = df[split_index + margin:]

        return train_data, test_data

    def _preprocess_data(self, X):
        # Normalize numerical features
        normalized_data = []
        for feature in self.numerical_features:
            if feature not in self.scalers:
                self.scalers[feature] = MinMaxScaler(feature_range=(0, 1))
                normalized_data.append(self.scalers[feature].fit_transform(X[feature].values.reshape(-1, 1)))
            else:
                normalized_data.append(self.scalers[feature].transform(X[feature].values.reshape(-1, 1)))

        # Convert categorical features to one-hot encoding
        onehot_encoder = OneHotEncoder(sparse=False)
        encoded_data = onehot_encoder.fit_transform(X[self.categorical_features])

        return np.concatenate(normalized_data + [encoded_data], axis=1)
