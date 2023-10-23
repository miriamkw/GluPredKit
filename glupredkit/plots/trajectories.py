from .base_plot import BasePlot
import numpy as np
import matplotlib.pyplot as plt
from glupredkit.helpers.unit_config_manager import config_manager


class Plot(BasePlot):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

    def __call__(self, models_data, y_true):
        def on_hover(event):
            if event.inaxes == ax:
                for line in lines:
                    contains, _ = line.contains(event)
                    if contains:
                        line.set_alpha(1.0)
                    else:
                        line.set_alpha(0.2)
                fig.canvas.draw_idle()

        prediction_index = int(self.prediction_horizon / 5)
        total_time = len(y_true) * 5 + prediction_index * 5
        t = np.arange(0, total_time, 5)

        if config_manager.use_mgdl:
            unit = "mg/dL"
        else:
            y_true = [config_manager.convert_value(val) for val in y_true]
            unit = "mmol/L"

        for model_data in models_data:
            fig, ax = plt.subplots()

            model_name = model_data.get('name')
            y_pred = model_data.get('y_pred')

            # Use correct unit
            if config_manager.use_mgdl:
                ax.axhspan(70, 180, facecolor='blue', alpha=0.2)
            else:
                y_pred = [config_manager.convert_value(val) for val in y_pred]
                ax.axhspan(config_manager.convert_value(70), config_manager.convert_value(180), facecolor='blue', alpha=0.2)

            ax.set_title('Blood glucose predicted trajectories')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(f'Blood glucose [{unit}]')
            ax.scatter(t[:len(y_true)], y_true, label='Blood glucose measurements', color='black')

            lines = []
            # Add predicted trajectories
            for i in range(0, len(y_pred) - prediction_index):
                line, = ax.plot([t[i], t[i + prediction_index]], [y_true[i], y_pred[i]], linestyle='--')
                lines.append(line)

            fig.canvas.mpl_connect('motion_notify_event', on_hover)
            ax.legend()
            plt.title(f'Predicted trajectories {self.prediction_horizon} Minutes Ahead for {model_name}')

            file_path = "data/figures/"
            file_name = f'trajectories_ph-{self.prediction_horizon}_{model_name}.png'
            plt.savefig(file_path + file_name)
            plt.show()

