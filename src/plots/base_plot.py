class BasePlot:
    def __init__(self, prediction_horizon):
        self.prediction_horizon = prediction_horizon

    def __call__(self, models_data, y_true):
        """
        Draw a plot for the given trained_models data.

        models_data: A list of dictionaries containing the model name, y_true, and y_pred.
                    Example: [{'name': 'model1', 'y_pred': [...]}, ...]
        y_true: A list of true measured values
        """

        raise NotImplementedError("BasePlot not implemented!")

