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
        Process the input dataset and split it into training and testing subsets.

        Args:
            df (DataFrame):
                The input dataset. It must include the following required columns:
                - `CGM` (float): Continuous Glucose Monitoring values.
                - `is_test` (bool): A flag indicating whether a row belongs to the test set.
                - `id` (int or str): A unique identifier for each individual.
                Optional columns:
                Additional features may be included based on configuration.

            **kwargs:
                Additional keyword arguments for configuration, such as:
                - `scaling` (bool): Whether to scale numeric features (default: False).
                - `feature_selection` (list): A list of features to retain for training/testing (default: None).

        Returns:
            tuple: A tuple containing:
                - train_data (DataFrame): The subset of the dataset to be used for model training, containing all rows where `is_test` is 0.
                - test_data (DataFrame): The subset of the dataset to be used for model testing, containing all rows where `is_test` is 1.
        """
        raise NotImplementedError("Preprocessor not implemented!")

    def __repr__(self):
        return self.__class__.__name__
