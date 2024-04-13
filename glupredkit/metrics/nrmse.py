from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager

class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Normalized RMSE')

    def __call__(self, y_true, y_pred, x):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
        
        # Normalization by Mean
        y_mean = np.mean(y_true)
        rmse_mean_normalized = self.normalize_rmse_by_mean(rmse, y_mean)

        # Normalization by Difference Between Maximum and Minimum
        y_max = np.max(y_true)
        y_min = np.min(y_true)
        rmse_difference_normalized = self.normalize_rmse_by_difference(rmse, y_max, y_min)

        # Normalization by Standard Deviation
        y_std = np.std(y_true)
        rmse_std_normalized = self.normalize_rmse_by_std(rmse, y_std)

        # Normalization by Interquartile Range
        q1, q3 = np.percentile(y_true, [25, 75])
        rmse_iqr_normalized = self.normalize_rmse_by_iqr(rmse, q1, q3)

        if unit_config_manager.use_mgdl:
            return {
                "RMSE_Mean_Normalized": rmse_mean_normalized,
                "RMSE_Difference_Normalized": rmse_difference_normalized,
                "RMSE_Std_Normalized": rmse_std_normalized,
                "RMSE_IQR_Normalized": rmse_iqr_normalized
            }
        else:
            return {
                "RMSE_Mean_Normalized": unit_config_manager.convert_value(rmse_mean_normalized),
                "RMSE_Difference_Normalized": unit_config_manager.convert_value(rmse_difference_normalized),
                "RMSE_Std_Normalized": unit_config_manager.convert_value(rmse_std_normalized),
                "RMSE_IQR_Normalized": unit_config_manager.convert_value(rmse_iqr_normalized)
            }

    def normalize_rmse_by_mean(self, rmse, y_mean):
        return rmse / y_mean
    
    def normalize_rmse_by_difference(self, rmse, y_max, y_min):
        return rmse / (y_max - y_min)
    
    def normalize_rmse_by_std(self, rmse, y_std):
        return rmse / y_std
    
    def normalize_rmse_by_iqr(self, rmse, q1, q3):
        return rmse / (q3 - q1)
