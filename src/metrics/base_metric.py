from typing import List

class BaseMetric:
    def __init__(self, name):
        self.name = name

    def __call__(self, y_true: List[float], y_pred: List[float]) -> float:
        raise NotImplementedError("Metric not implemented!")

    def __repr__(self):
        return self.name