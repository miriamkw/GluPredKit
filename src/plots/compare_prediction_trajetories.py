from src.plots.base_plot import BasePlot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class ComparePredictionTrajectories(BasePlot):

    def __call__(self, models, model_names, df, measurements=None, interval=5, use_mgdl=True, history_samples=36):
        """
        A plot of a prediction, where you can change the carbohydrate and insulin input with a slider, and observe
        how the model reacts.

        models -- a list of prediction models implementing the base_model
        model_names -- a list of model_names corresponding to the models
        df -- a dataframe that the prediction model can use as input to get a prediction
        interval -- the interval between each prediction in the trajectory
        use_mgdl -- whether to present results in mg/dL or mmol/L
        history_samples -- how much previous blood glucose samples to include in the plot
        """
        if use_mgdl:
            k = 1
            unit_label = 'mg/dL'
        else:
            k = 18.0182
            unit_label = 'mmol/L'

        font = {
            'size': 18,
        }

        plt.rc('font', **font)

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)

        for i in range(len(models)):
            y_pred = models[i].predict(df)
            n_predictions = len(y_pred[0])
            n_intervals = int(interval / 5)  # Intervals from minutes to number of elements
            t = np.arange(-history_samples * 5 + 5, (n_predictions * n_intervals) * 5, 5)

            ax.plot(t[history_samples - 1:n_intervals * n_predictions + history_samples:n_intervals],
                    [el / k for el in y_pred[-1]], linestyle='--', label=model_names[i])

        non_zero_insulin_t = [x for x, y in zip(t[:history_samples], df['insulin'][-history_samples:]) if y != 0]
        non_zero_insulin = [y * 18.0182 / k for y in df['insulin'][-history_samples:] if y != 0]
        for i in range(len(non_zero_insulin)):
            if non_zero_insulin[i] < 1:
                text = ''
            else:
                text = f'{non_zero_insulin[i] * k / 18.0182:.1f} IU'
            plt.text(non_zero_insulin_t[i], non_zero_insulin[i], text, ha='center', va='bottom')

        non_zero_carbs_t = [x for x, y in zip(t[:history_samples], df['carbs'][-history_samples:]) if y != 0]
        non_zero_carbs = [y / k * 2 for y in df['carbs'][-history_samples:] if y != 0]
        for i in range(len(non_zero_carbs)):
            text = f'{non_zero_carbs[i] * k / 2:.0f} g'
            plt.text(non_zero_carbs_t[i], non_zero_carbs[i], text, ha='center', va='bottom')

        ax.scatter(t[:history_samples], [el/k for el in df['CGM'][-history_samples:]], label='Blood glucose measurements', color='black')
        ax.scatter(non_zero_insulin_t, non_zero_insulin, label='Insulin')
        ax.scatter(non_zero_carbs_t, non_zero_carbs, label='Carbohydrates')

        if measurements:
            n_measurement = len(measurements)
            ax.scatter(t[history_samples:history_samples + n_measurement], measurements, color='black')

        ax.axhspan(70 / k, 180 / k, facecolor='blue', alpha=0.2)

        plt.xlabel('Time [min]')
        plt.ylabel('Blood Glucose [' + unit_label + ']')

        plt.legend(loc='lower right')

        plt.show()