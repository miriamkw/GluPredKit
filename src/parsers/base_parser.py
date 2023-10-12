class BaseParser:
    def __init__(self):
        pass

    def __call__(self, start_date, end_date, username, password, **kwargs):
        """
        start_date -- start date for dataset
        end_date -- end date for dataset
        optional additional parameters such as api username and key

        Returns four dataframes on the format specified in the README.
        """
        raise NotImplementedError("Metric not implemented!")

    def __repr__(self):
        return self.__class__.__name__
