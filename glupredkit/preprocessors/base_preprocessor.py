"""
All preprocessors must:
- Handle feature addition
- Handle feature imputation
- Add a target column named "target"
- Drop nan-values
- Split data into train- and testdata
"""


class BasePreprocessor:
    def __init__(self, numerical_features, categorical_features, what_if_features, prediction_horizon,
                 num_lagged_features, test_size):
        """
        Args:
            numerical_features (list of strings): The names of the numerical features in the processed data
            categorical_features (list of strings): The names of the categorical features in the processed data
            what_if_features (list of strings): The names of the what-if features in the processed data
            prediction_horizon (int): The prediction horizon in minutes.
            num_lagged_features (int): The number of time-lagged features to generate (12 samples corresponds to one
            hour).
            test_size (float): The fraction of data to reserve for testing.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.what_if_features = what_if_features
        self.prediction_horizon = prediction_horizon
        self.num_lagged_features = num_lagged_features
        self.test_size = test_size

    def __call__(self, data, **kwargs):
        """
        Args:
            data (DataFrame): The input dataset.

        Returns:
            train_data: The dataset for model training.
            test_data: The dataset for model testing.
        """
        raise NotImplementedError("Preprocessor not implemented!")

    def __repr__(self):
        return self.__class__.__name__
