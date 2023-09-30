class BasePreprocessor:
    def __init__(self):
        pass

    def __call__(self, data, **kwargs):
        """
        data -- raw data input
        optional additional parameters

        Returns two dataframes: The training data and the test data.
        """
        raise NotImplementedError("Preprocessor not implemented!")

    def __repr__(self):
        return self.__class__.__name__
