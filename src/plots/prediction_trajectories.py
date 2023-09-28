from src.plots.base_plot import BasePlot
import numpy as np
import matplotlib.pyplot as plt


class PredictionTrajectories(BasePlot):
    def __init__(self):
        super().__init__()

    def _draw_plot(self, y_pred, interval=5, use_mgdl=True):
        """
        It is assumed that y_pred is either a list of predicted values, or a list of lists of predicted trajectories.
        In case of only one predicted value, output_offset cannot be None.
        This function assumes that the CGM measurements have a 5-minute interval.

        y_pred -- a list of predicted trajectories, where the first value is the referenced measurement
        interval -- the interval in minutes between each prediction in the trajetories.
        """
        if use_mgdl:
            k = 1
        else:
            k = 18.0182

        font = {
            'size': 18,
        }

        plt.rc('font', **font)

        def on_hover(event):
            if event.inaxes == ax:
                for line in lines:
                    contains, _ = line.contains(event)
                    if contains:
                        line.set_alpha(1.0)
                    else:
                        #line.set_alpha(0.2)
                        line.set_alpha(1.0)
                fig.canvas.draw_idle()

        total_time = len(y_pred) * 5 + 12 * 6 * 5
        t = np.arange(0, total_time, 5)
        measurements = [trajectory[0] for trajectory in y_pred]

        fig, ax = plt.subplots()

        ax.set_title('OTS RR - Predicted Blood Glucose Trajectories')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Blood glucose (mg/dL)')
        ax.scatter(t[:len(y_pred)], [el/k for el in measurements], label='Blood glucose measurements', color='black')
        ax.axhspan(70/k, 180/k, facecolor='blue', alpha=0.2)

        n_predictions = len(y_pred[0])
        n_intervals = int(interval / 5)  # Intervals from minutes to number of elements

        lines = []

        # Add predicted trajectories
        for i in range(0, len(y_pred)):
            # Remove predictions where we don't have the reference measurement available
            x = t[i:min(n_intervals * n_predictions + i, len(measurements)):n_intervals]
            y = [el/k for el in y_pred[i]][:len(x)]

            line, = ax.plot(x, y, linestyle='--')
            lines.append(line)

        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        ax.legend()
        plt.show()
