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

    def __call__(self, dfs, *args):
        """
        This plot plots predicted trajectories from the measured values. A random subsample of around 24 hours will
        be plotted.
        """
        def on_hover(event):
            if event.inaxes == ax:
                for hover_line in lines:
                    contains, _ = hover_line.contains(event)
                    if contains:
                        hover_line.set_alpha(1.0)
                    else:
                        hover_line.set_alpha(0.2)
                fig.canvas.draw_idle()

        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
        else:
            unit = "mmol/L"

        for df in dfs:
            fig, ax = plt.subplots()

            model_name = df['Model Name'][0]
            ph = int(df['prediction_horizon'][0])

            prediction_horizons = range(5, ph + 1, 5)

            y_true = df[f'target_5'][0]
            y_true = ast.literal_eval(y_true)

            n_samples = 12*24

            if len(y_true) - n_samples > len(y_true):
                start_index = random.randint(0, len(y_true) - n_samples)
            else:
                start_index = 0

            y_pred_lists = []

            for ph in prediction_horizons:
                y_pred = df[f'y_pred_{ph}'][0]
                y_pred = y_pred.replace("nan", "None")
                y_pred = ast.literal_eval(y_pred)
                y_pred = [np.nan if val is None else val for val in y_pred]

                y_pred_lists += [y_pred]

            y_true = np.array(y_true)[start_index:start_index + n_samples]
            y_pred_lists = np.array(y_pred_lists)
            y_pred_lists = np.transpose(y_pred_lists)[1 + start_index:start_index + n_samples + 1]

            total_time = len(y_true) * 5 + ph
            t = np.arange(0, total_time, 5)

            # Use correct unit
            if unit_config_manager.use_mgdl:
                ax.axhspan(70, 180, facecolor='blue', alpha=0.2)
            else:
                y_true = [unit_config_manager.convert_value(val) for val in y_true]
                ax.axhspan(unit_config_manager.convert_value(70), unit_config_manager.convert_value(180),
                           facecolor='blue', alpha=0.2)

            ax.set_title('Blood glucose predicted trajectories')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(f'Blood glucose [{unit}]')
            ax.scatter(t[:len(y_true)], y_true, label='Blood glucose measurements', color='black')

            prediction_horizons = range(0, ph + 1, 5)
            lines = []
            # Add predicted trajectories
            for i in range(len(y_pred_lists)):
                # Adding the true measurement to the trajectory
                trajectory = np.insert(y_pred_lists[i], 0, y_true[i])
                line, = ax.plot([val + i*5 for val in prediction_horizons], trajectory, linestyle='--')
                lines.append(line)

            fig.canvas.mpl_connect('motion_notify_event', on_hover)
            ax.legend()
            plt.title(f'Predicted trajectories for {model_name}')
            file_path = "data/figures/"

            timestamp = datetime.now().isoformat()
            safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
            safe_timestamp = safe_timestamp.replace('.', '_')
            file_name = f"trajectories_{model_name}_{safe_timestamp}.png"
            plt.savefig(file_path + file_name)
            plt.show()
