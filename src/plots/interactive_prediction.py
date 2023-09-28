from src.plots.base_plot import BasePlot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import datetime
import pytz
import pandas as pd


class InteractivePrediction(BasePlot):

    def __call__(self, model, df, interval=5, use_mgdl=True, history_samples=36):
        """
        A plot of a prediction, where you can change the carbohydrate and insulin input with a slider, and observe
        how the model reacts.

        model -- a prediction model implementing the base_model
        df -- a dataframe that the prediction model can use as input to get a prediction
        interval -- the interval between each prediction in the trajectory
        use_mgdl -- whether to present results in mg/dL or mmol/L
        history_samples -- how much previous blood glucose samples to include in the plot
        """
        if use_mgdl:
            k = 1
        else:
            k = 18.0182

        y_pred = model.predict(df)
        n_predictions = len(y_pred[0])
        n_intervals = int(interval / 5)  # Intervals from minutes to number of elements

        t = np.arange(-history_samples * 5 + 5, (n_predictions * n_intervals) * 5, 5)

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)

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

        l_measurements = ax.scatter(t[:history_samples], [el/k for el in df['CGM'][-history_samples:]], label='Blood glucose measurements')
        l_insulin = ax.scatter(non_zero_insulin_t, non_zero_insulin, label='Insulin')
        l_carbohydrates = ax.scatter(non_zero_carbs_t, non_zero_carbs, label='Carbohydrates')

        l2, = ax.plot(t[history_samples - 1:n_intervals * n_predictions + history_samples:n_intervals], [el/k for el in y_pred[-1]], linestyle='--', label='Blood glucose predictions')

        ax_insulin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        ax_carbs = fig.add_axes([0.25, 0.15, 0.65, 0.03])

        # create the sliders
        s_insulin = Slider(
            ax_insulin, "Insulin", 0.0, 10.0,
            valinit=0, valstep=1,
            color="green"
        )

        s_carbs = Slider(
            ax_carbs, "Carbohydrates", 0.0, 100.0,
            valinit=0, valstep=10,
            initcolor='none'  # Remove the line marking the valinit position.
        )

        def update(val):
            insulin = s_insulin.val
            carbs = s_carbs.val

            last_row_index = df.index[-1]

            df.at[last_row_index, 'carbs'] = carbs
            df.at[last_row_index, 'insulin'] = insulin

            y_pred_new = model.predict(df)
            l2.set_ydata([el/k for el in y_pred_new[-1]])
            fig.canvas.draw_idle()

        s_insulin.on_changed(update)
        s_carbs.on_changed(update)

        ax.axhspan(70 / k, 180 / k, facecolor='blue', alpha=0.2)

        ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(ax_reset, 'Reset', hovercolor='0.975')

        def reset(event):
            s_insulin.reset()
            s_carbs.reset()

        button.on_clicked(reset)

        plt.legend(handles=[l_measurements, l2, l_insulin, l_carbohydrates], bbox_to_anchor=(-0.6, 10), loc='upper left')

        plt.show()


