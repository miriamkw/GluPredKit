from typing import List

class BasePlot:
    def __init__(self, name):
        pass

    def __call__(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        It is expected that y_pred is either a list of predicted values, or a list of lists of predicted trajectories
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        try:
            return self._draw_plot(y_true, y_pred)
        except NotImplementedError:
            raise NotImplementedError("Metric not implemented!")

    def _draw_plot(self, y_true: List[float], y_pred: List[float]) -> float:
        raise NotImplementedError("Metric not implemented!")
