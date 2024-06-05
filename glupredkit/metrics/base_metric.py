from typing import List
from abc import ABC, abstractmethod
import warnings
import numpy as np


class BaseMetric(ABC):
    def __init__(self, name):
        self.name = name

    def __call__(self, y_true: List[float], y_pred: List[float]) -> any:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        # Check for NaN values and issue a warning if found
        if any(np.isnan(y) for y in y_true) or any(np.isnan(y) for y in y_pred):
            warnings.warn("NaN values detected in metric input", UserWarning)

        # Check for zero or negative values in y_true
        if any(y <= 0 for y in y_true):
            raise ValueError("y_true contains zero or negative values")

        try:
            return self._calculate_metric(y_true, y_pred)

        except NotImplementedError:
            raise NotImplementedError("Metric not implemented!")

    @abstractmethod
    def _calculate_metric(self, y_true: List[float], y_pred: List[float]) -> any:
        raise NotImplementedError("Metric not implemented!")

    def __repr__(self):
        return self.name
