"""
All preprocessors must:
- Handle feature addition
- Handle feature imputation
- Add a target columns starting with "target"
"""

class BasePreprocessor:
    def __init__(self, subject_ids, numerical_features, categorical_features, what_if_features, prediction_horizon,
                 num_lagged_features):
        """
        Args:
            prediction_horizon (int): The prediction horizon in minutes.
            num_lagged_features (int): The number of time-lagged features to generate (12 samples corresponds to one
            hour).
        """
        self.subject_ids = subject_ids
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.what_if_features = what_if_features
        self.prediction_horizon = prediction_horizon
        self.num_lagged_features = num_lagged_features

    def __call__(self, df, **kwargs):
        """
        Args:
            df (DataFrame): The input dataset.

        Returns:
            train_data: The dataset for model training.
            test_data: The dataset for model testing.
        """
        raise NotImplementedError("Preprocessor not implemented!")

    def __repr__(self):
        return self.__class__.__name__
