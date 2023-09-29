class BasePreprocessor:
    def __init__(self, name):
        self.name = name

    def __call__(self, data, **kwargs):
        """
        start_date -- start date for dataset
        end_date -- end date for dataset
        optional additional parameters such as api username and key

        Returns four dataframes on the format specified in the README.
        """
        raise NotImplementedError("Metric not implemented!")

    def __repr__(self):
        return self.__class__.__name__
