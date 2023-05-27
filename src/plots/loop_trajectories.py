from src.plots.base_plot import BasePlot
import matplotlib.pyplot as plt
import datetime, pytz
from pyloopkit.loop_math import predict_glucose

class LoopTrajectories(BasePlot):

    def __init__(self):
        super().__init__()

    def _draw_plot(self, loop_model_output, glucose_unit='mg/dL'):

        if glucose_unit == 'mg/dL':
            unit_const = 1
        elif glucose_unit == 'mmol/L':
            unit_const = 18.0182
        else:
            raise Exception("Invalid glucose unit: " + glucose_unit + ", has to be either mg/dL or mmol/L")

        inputs = loop_model_output.get("input_data")

        glucose_date_inputs = inputs.get("glucose_dates")[-12:]
        glucose_value_inputs = inputs.get("glucose_values")[-12:]
        glucose_value_inputs = [val / unit_const for val in glucose_value_inputs]

        glucose_dates = loop_model_output.get("predicted_glucose_dates")[:73]
        glucose_values = [val / unit_const for val in loop_model_output.get("predicted_glucose_values")[:73]]

        start_date = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
        start_glucose = inputs.get("glucose_values")[-1]

        (carb_dates,
         carb_values
         ) = predict_glucose(
            start_date, start_glucose,
            carb_effect_dates=loop_model_output.get("carb_effect_dates"),
            carb_effect_values=loop_model_output.get("carb_effect_values")
        )
        carb_values = [val / unit_const for val in carb_values]

        (insulin_dates,
         insulin_values
         ) = predict_glucose(
            start_date, start_glucose,
            insulin_effect_dates=loop_model_output.get("insulin_effect_dates"),
            insulin_effect_values=loop_model_output.get("insulin_effect_values")
        )
        insulin_values = [val / unit_const for val in insulin_values]

        (momentum_dates,
         momentum_values
         ) = predict_glucose(
            start_date, start_glucose,
            momentum_dates=loop_model_output.get("momentum_effect_dates"),
            momentum_values=loop_model_output.get("momentum_effect_values")
        )
        momentum_values = [val / unit_const for val in momentum_values]

        if loop_model_output.get("retrospective_effect_dates"):
            (retrospective_dates,
             retrospective_values
             ) = predict_glucose(
                start_date, start_glucose,
                correction_effect_dates=loop_model_output.get(
                    "retrospective_effect_dates"
                ),
                correction_effect_values=loop_model_output.get(
                    "retrospective_effect_values"
                )
            )
        else:
            (retrospective_dates,
             retrospective_values
             ) = ([], [])
        retrospective_values = [val / unit_const for val in retrospective_values]

        print("Start glucose: ", start_glucose / unit_const)
        print("Final prediction: ", glucose_values[-1])
        print("")
        if not len(insulin_values) == 0:
            print("Final insulin: ", insulin_values[-1])
        if not len(carb_values) == 0:
            print("Final carbs: ", carb_values[-1])
        if not len(momentum_values) == 0:
            print("Final momentum: ", momentum_values[-1])
        if not len(retrospective_values) == 0:
            print("Final retrospective: ", retrospective_values[-1])

        fig, ax = plt.subplots()
        ax.scatter(glucose_date_inputs, glucose_value_inputs, color='blue', label='True')
        ax.plot(glucose_dates, glucose_values, linestyle='--', color='blue', label='Predicted')
        ax.plot(carb_dates[:73], carb_values[:73], label='Carbohydrates', linestyle='--', color='orange')
        ax.plot(insulin_dates[:73], insulin_values[:73], label='Insulin', linestyle='--')
        ax.plot(momentum_dates, momentum_values, label='Momentum', linestyle='--')
        if len(retrospective_values) == 0:
            print("No retrospective values")
        else:
            ax.plot(retrospective_dates, retrospective_values, label='Retrospective', linestyle='--')

        plt.axhspan(100 / unit_const, 113 / unit_const, facecolor='b', alpha=0.2)

        ax.set(xlabel='Time (minutes)', ylabel='Blood Glucose (' + glucose_unit + ')',
               title='Measured vs one trajectory of predicted values')
        ax.grid()
        plt.legend(loc='best')

        plt.show()

