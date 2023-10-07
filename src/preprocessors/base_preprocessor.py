class BasePreprocessor:
    def __init__(self):
        pass

    def __call__(self, data, prediction_horizon, num_lagged_features, include_hour, test_size, **kwargs):
        """
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
        raise NotImplementedError("Preprocessor not implemented!")

    def __repr__(self):
        return self.__class__.__name__
