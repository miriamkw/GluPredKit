class BasePlot:
    def __init__(self, name):
        pass

    def __call__(self):
        """
        It is expected that y_pred is either a list of predicted values, or a list of lists of predicted trajectories
        """
        raise NotImplementedError("Metric not implemented!")

