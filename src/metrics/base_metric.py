class BaseMetric:
    def __init__(self, name):
        self.name = name

    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Metric not implemented!")

    def __repr__(self):
        return self.name