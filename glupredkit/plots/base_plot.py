class BasePlot:
    def __init__(self):
        pass

    def __call__(self, dfs, *args, **kwargs):
        """
        Draw a plot for the given trained_models data.

        dfs: A list of dataframes containing the results from the tested model.
                    Example: [{'name': 'model1', 'y_pred': [...]}, ...]
        """

        raise NotImplementedError("BasePlot not implemented!")

