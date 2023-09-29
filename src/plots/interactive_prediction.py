from base_plot import BasePlot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import datetime
import pytz
import pandas as pd

class InteractivePrediction(BasePlot):
    def __init__(self):
        super().__init__('MAE')

    def __call__(self, model, df_glucose, df_bolus, df_basal, df_carbs):
        """
        It is expected that y_pred is one predicted trajectory
        In case of only one predicted value, output_offset cannot be None.
        """
        # To do to start: predict one trajectory and predicted
        t = np.arange(0.0, (12 + 72)*5, 5)
        y_pred = model.predict_one_prediction(df_glucose, df_bolus, df_basal, df_carbs)
        if df_glucose['units'][0] == 'mmol/L':
            glucose_values = [value * 18.0182 for value in df_glucose['value'][:12].to_numpy()[::-1]]
        else:
            glucose_values = df_glucose['value'][:12].to_numpy()[::-1]

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)
        l = ax.scatter(t[:12], glucose_values)
        l2, = ax.plot(t[12:], y_pred, linestyle='--')

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
            time = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)

            new_row_insulin = pd.DataFrame({'time': time, 'dose[IU]': insulin}, index=[0])
            new_row_carbs = pd.DataFrame({'time': time, 'units': 'grams', 'value': carbs, 'absorption_time[s]': 10800}, index=[0])

            df_carbs_copy = pd.concat([df_carbs, new_row_carbs], axis=0)
            df_bolus_copy = pd.concat([df_bolus, new_row_insulin], axis=0)

            df_carbs_copy.reset_index(inplace=True)
            df_bolus_copy.reset_index(inplace=True)
            y_pred = model.predict_one_prediction(df_glucose,
                                                  df_bolus_copy,
                                                  df_basal,
                                                  df_carbs_copy)
            l2.set_ydata(y_pred)
            fig.canvas.draw_idle()

        s_insulin.on_changed(update)
        s_carbs.on_changed(update)

        ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(ax_reset, 'Reset', hovercolor='0.975')

        def reset(event):
            s_insulin.reset()
            s_carbs.reset()

        button.on_clicked(reset)

        plt.show()
