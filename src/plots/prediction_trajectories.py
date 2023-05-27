from src.plots.base_plot import BasePlot
import numpy as np
import matplotlib.pyplot as plt

class PredictionTrajectories(BasePlot):
    def __init__(self):
        super().__init__()

    def _draw_plot(self, y_true, y_pred, output_offset=None):
        """
        It is expected that y_pred is either a list of predicted values, or a list of lists of predicted trajectories.
        In case of only one predicted value, output_offset cannot be None.
        """
        def on_hover(event):
            if event.inaxes == ax:
                for line in lines:
                    contains, _ = line.contains(event)
                    if contains:
                        line.set_alpha(1.0)
                    else:
                        line.set_alpha(0.2)
                fig.canvas.draw_idle()

        total_time = len(y_true)*5 + 12*6*5
        t = np.arange(0, total_time, 5)
        measurements = [trajectory[0] for trajectory in y_true]

        fig, ax = plt.subplots()

        ax.set_title('Blood glucose predicted trajectories')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Blood glucose (mg/dL)')
        ax.scatter(t[:len(y_true)], measurements, label='Blood glucose measurements', color='black')
        ax.axhspan(70, 180, facecolor='blue', alpha=0.2)

        n_predictions = len(y_pred[0])
        lines = []
        # Add predicted trajectories
        for i in range(0, len(y_pred)):
            line, = ax.plot(t[i:n_predictions + i], y_pred[i], linestyle='--')
            lines.append(line)

        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        ax.legend()
        plt.show()
