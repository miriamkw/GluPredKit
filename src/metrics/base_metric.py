from typing import List

class BaseMetric:
    def __init__(self, name):
        self.name = name
    """
    def __call__(self, y_true: List[float], y_pred: List[float]) -> float:
        raise NotImplementedError("Metric not implemented!")
    """
    def __call__(self, y_true: List[float], y_pred: List[float]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        try:
            return self._calculate_metric(y_true, y_pred)
        except NotImplementedError:
            raise NotImplementedError("Metric not implemented!")

    def _calculate_metric(self, y_true: List[float], y_pred: List[float]) -> float:
        raise NotImplementedError("Metric not implemented!")

    def __repr__(self):
        return self.name