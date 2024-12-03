from .base_plot import BasePlot
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
from glupredkit.helpers.unit_config_manager import unit_config_manager
from datetime import datetime


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, col='CGM', *args):
        """
        This plot plots predicted trajectories from the measured values. A random subsample of around 24 hours will
        be plotted.
        """

        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
        else:
            unit = "mmol/L"

        for df in dfs:
            fig, ax = plt.subplots()

            y_true = df[f'target_5'][0]
            string_values = y_true.replace("nan", "None")
            y_true = ast.literal_eval(string_values)
            y_true = [np.nan if x is None else x for x in y_true]

            hypo_threshold = 70
            hyper_threshold = 180
            if not unit_config_manager.use_mgdl:
                y_true = [unit_config_manager.convert_value(val) for val in y_true]
                hypo_threshold = unit_config_manager.convert_value(hypo_threshold)
                hyper_threshold = unit_config_manager.convert_value(hyper_threshold)

            ax.hist(y_true, bins=20, edgecolor='black')

            # Add vertical lines for hypo- and hyperglycemic thresholds
            ax.axvline(hypo_threshold, color='red', linestyle='dashed', linewidth=2,
                       label=f'Glycemic threshold')
            ax.axvline(hyper_threshold, color='red', linestyle='dashed', linewidth=2)

            # Set title and labels
            ax.set_title(f'Histogram of target {col} values')
            ax.set_xlabel(f'Blood glucose [{unit}]')
            ax.set_ylabel('Frequency')
            ax.legend()

            plt.show()

