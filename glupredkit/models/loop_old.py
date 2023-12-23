from glupredkit.models.base_model import BaseModel
import datetime
import pandas as pd
import pytz

from pyloopkit.dose import DoseType
from pyloopkit.generate_graphs import plot_graph, plot_loop_inspired_glucose_graph
#from .loop_kit_tests import find_root_path
from pyloopkit.loop_math import predict_glucose
from pyloopkit.pyloop_parser import (
    parse_report_and_run, parse_dictionary_from_previous_run
)
from pyloopkit.loop_data_manager import update


class Model(BaseModel):

    # TODO: Add inputs for user-defined adjustments, like DIA, insulin absorbption model ...


    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

    def fit(self, x_train, y_train):
        # Manually adjust the therapy settings at the bottom of this file to adjust the prediction model

        # TODO: Should there be a therapy settings dict as input here?

        return self

    def predict(self, x_test):
        """
        Return:
        y_pred -- A list of lists of the predicted trajectories.
        """

        # TODO: Accounting for time zones in the predictions

        input_dict = self.get_input_dict()

        dose_types, start_times, end_times, dose_values, dose_delivered_units = self.get_insulin_data(x_test['insulin'],
                                                                                                      None)
        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = start_times
        input_dict["dose_end_times"] = end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = dose_delivered_units

        carb_dates, carb_values, carb_absorption_times = self.get_carbohydrate_data(x_test['carbs'])
        input_dict["carb_dates"] = carb_dates
        input_dict["carb_values"] = carb_values
        input_dict["carb_absorption_times"] = carb_absorption_times

        history_length = 6
        n_predictions = x_test['CGM'].shape[0] - 12*6 - history_length

        if n_predictions <= 0:
            print("Not enough data to predict.")
            return

        # Sort glucose values
        df_glucose = x_test['CGM'].sort_values(by='time', ascending=True).reset_index(drop=True)

        # For each glucose measurement, get a new prediction
        # Skip iteration if there are not enough predictions.
        y_pred = []
        for i in range(history_length, n_predictions + history_length):
            time_to_calculate = df_glucose["time"][i]
            input_dict["time_to_calculate_at"] = time_to_calculate

            # NOTE: Not filtering away future glucose values will lead to erroneous prediction results!
            glucose_dates, glucose_values = self.get_glucose_data(df_glucose[df_glucose['time'] < time_to_calculate])
            input_dict["glucose_dates"] = glucose_dates
            input_dict["glucose_values"] = glucose_values

            output_dict = update(input_dict)

            if len(output_dict.get("predicted_glucose_values")) < 73:
                print("Not enough predictions. Skipping iteration...")
                continue
            else:
                # Starting from index 1 to skip the reference value in the predicted trajectory
                y_pred.append(output_dict.get("predicted_glucose_values")[1:73])
        return y_pred



    def one_prediction_testing(self, df):
        # Important docs: https://github.com/miriamkw/PyLoopKit/blob/develop/pyloopkit/docs/pyloopkit_documentation.md
        # For correct predictions, at least 24 hours + duration of insulin absorption (DIA) of data is needed
        DIA = 60*6

        # TODO: For all trajectories, this should dynamically change with the index
        # NOTE: Not filtering away future glucose values will lead to erroneous prediction results!
        #time_to_calculate = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
        time_to_calculate = df.index[-1]
        filtered_df = df[df.index <= time_to_calculate].tail(12*24 + int(DIA / 5))

        input_dict = self.get_input_dict()
        input_dict["time_to_calculate_at"] = time_to_calculate

        input_dict["glucose_dates"] = filtered_df.index.tolist()
        input_dict["glucose_values"] = filtered_df.CGM.tolist()

        # TODO: Get the dose type in parser, basal temp vs normal!
        # TODO: Add bolus doses as well
        input_dict["dose_types"] = [DoseType.from_str("bolus") for _ in filtered_df.index.tolist()]
        input_dict["dose_start_times"] = filtered_df.index.tolist()
        input_dict["dose_end_times"] = filtered_df.index.tolist()
        input_dict["dose_values"] = filtered_df.basal.tolist()
        input_dict["dose_delivered_units"] = [None for _ in filtered_df.index.tolist()]

        input_dict["carb_dates"] = filtered_df[filtered_df.carbs > 0].index.tolist()
        input_dict["carb_values"] = filtered_df[filtered_df.carbs > 0].carbs.tolist()
        # TODO: Add proper carb absorption times
        input_dict["carb_absorption_times"] = [360 for _ in filtered_df[filtered_df.carbs > 0].index.tolist()]

        output_dict = update(input_dict)

        #print(output_dict.get("predicted_glucose_values"))
        #print(output_dict.get("predicted_glucose_dates"))


        # TODO: Testing. plot data, observe it looks similar to iAPS (especially basal...)

        return output_dict



    def plot_prediction_contributions(self, df):

        recommendations = self.one_prediction_testing(df)

        # %% generate separate glucose predictions using each effect individually
        starting_date = recommendations.get("input_data").get("glucose_dates")[-1]
        starting_glucose = recommendations.get("input_data").get("glucose_values")[-1]

        (momentum_predicted_glucose_dates,
         momentum_predicted_glucose_values
         ) = predict_glucose(
            starting_date, starting_glucose,
            momentum_dates=recommendations.get("momentum_effect_dates"),
            momentum_values=recommendations.get("momentum_effect_values")
        )

        (insulin_predicted_glucose_dates,
         insulin_predicted_glucose_values
         ) = predict_glucose(
            starting_date, starting_glucose,
            insulin_effect_dates=recommendations.get("insulin_effect_dates"),
            insulin_effect_values=recommendations.get("insulin_effect_values")
        )

        (carb_predicted_glucose_dates,
         carb_predicted_glucose_values
         ) = predict_glucose(
            starting_date, starting_glucose,
            carb_effect_dates=recommendations.get("carb_effect_dates"),
            carb_effect_values=recommendations.get("carb_effect_values")
        )

        if recommendations.get("retrospective_effect_dates"):
            (retrospective_predicted_glucose_dates,
             retrospective_predicted_glucose_values
             ) = predict_glucose(
                starting_date, starting_glucose,
                correction_effect_dates=recommendations.get(
                    "retrospective_effect_dates"
                ),
                correction_effect_values=recommendations.get(
                    "retrospective_effect_values"
                )
            )
        else:
            (retrospective_predicted_glucose_dates,
             retrospective_predicted_glucose_values
             ) = ([], [])

        # plot insulin effects
        plot_graph(
            recommendations.get("insulin_effect_dates"),
            recommendations.get("insulin_effect_values"),
            title="Insulin Effect",
            grid=True,
        )


        # plot counteraction effects
        plot_graph(
            recommendations.get("counteraction_effect_start_times")[
            # trim to a reasonable length so the effects aren't too close together
            -len(recommendations.get("insulin_effect_dates")):
            ],
            recommendations.get("counteraction_effect_values")[
            # trim to a reasonable length so the effects aren't too close together
            -len(recommendations.get("insulin_effect_dates")):
            ],
            title="Counteraction Effects",
            fill_color="#f09a37",
            grid=True
        )

        # only plot carb effects if we have that data
        if recommendations.get("carb_effect_values"):
            plot_graph(
                recommendations.get("carb_effect_dates"),
                recommendations.get("carb_effect_values"),
                title="Carb Effect",
                line_color="#5FCB49",
                grid=True
            )

        # only plot the carbs on board over time if we have that data
        if recommendations.get("cob_timeline_values"):
            plot_graph(
                recommendations.get("cob_timeline_dates"),
                recommendations.get("cob_timeline_values"),
                title="Carbs on Board",
                line_color="#5FCB49", fill_color="#63ed47"
            )

        # %% Visualize output data as a Loop-style plot
        inputs = recommendations.get("input_data")

        plot_loop_inspired_glucose_graph(
            recommendations.get("predicted_glucose_dates"),
            recommendations.get("predicted_glucose_values"),
            title="Predicted Glucose",
            line_color="#5ac6fa",
            grid=True,
            previous_glucose_dates=inputs.get("glucose_dates")[-15:],
            previous_glucose_values=inputs.get("glucose_values")[-15:],
            correction_range_starts=inputs.get("target_range_start_times"),
            correction_range_ends=inputs.get("target_range_end_times"),
            correction_range_mins=inputs.get("target_range_minimum_values"),
            correction_range_maxes=inputs.get("target_range_maximum_values")
        )

        plot_loop_inspired_glucose_graph(
            recommendations.get("predicted_glucose_dates"),
            recommendations.get("predicted_glucose_values"),
            momentum_predicted_glucose_dates,
            momentum_predicted_glucose_values,
            insulin_predicted_glucose_dates,
            insulin_predicted_glucose_values,
            carb_predicted_glucose_dates,
            carb_predicted_glucose_values,
            retrospective_predicted_glucose_dates,
            retrospective_predicted_glucose_values,
            title="Predicted Glucose",
            line_color="#5ac6fa",
            grid=True,
            previous_glucose_dates=inputs.get("glucose_dates")[-15:],
            previous_glucose_values=inputs.get("glucose_values")[-15:],
            correction_range_starts=inputs.get("target_range_start_times"),
            correction_range_ends=inputs.get("target_range_end_times"),
            correction_range_mins=inputs.get("target_range_minimum_values"),
            correction_range_maxes=inputs.get("target_range_maximum_values")
        )

    def get_prediction_output(self, df_glucose, df_bolus, df_basal, df_carbs):
        """
        Get the prediction output dictionary from pyloopkit using the current time as the prediction time.
        """
        input_dict = self.get_input_dict()

        dose_types, start_times, end_times, dose_values, dose_delivered_units = self.get_insulin_data(df_bolus,
                                                                                                      df_basal)
        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = start_times
        input_dict["dose_end_times"] = end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = dose_delivered_units

        carb_dates, carb_values, carb_absorption_times = self.get_carbohydrate_data(df_carbs)
        input_dict["carb_dates"] = carb_dates
        input_dict["carb_values"] = carb_values
        input_dict["carb_absorption_times"] = carb_absorption_times

        # Sort glucose values
        df_glucose = df_glucose.sort_values(by='time', ascending=True).reset_index(drop=True)

        time_to_calculate = df_glucose["time"][-1]
        #time_to_calculate = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
        input_dict["time_to_calculate_at"] = time_to_calculate

        # NOTE: Not filtering away future glucose values will lead to erroneous prediction results!
        glucose_dates, glucose_values = self.get_glucose_data(df_glucose[df_glucose['time'] < time_to_calculate])
        input_dict["glucose_dates"] = glucose_dates
        input_dict["glucose_values"] = glucose_values

        return update(input_dict)

    def get_glucose_data(self, df):
        # Sort glucose values
        df = df.sort_values(by='time', ascending=True).reset_index(drop=True)
        glucose_dates = [timestamp for timestamp in df['time'].to_numpy()]
        glucose_values = df['value'].to_numpy()
        return glucose_dates, glucose_values

    def get_insulin_data(self, df_bolus, df_basal):
        df_bolus = df_bolus.copy()
        #df_basal = df_basal.copy()

        # Merge datasets:
        df_bolus['delivery_type'] = "bolus"
        df_bolus['duration[ms]'] = 0.0
        # Rename columns
        #df_bolus.rename(columns={"insulin": "values"}, inplace=True)
        #df_basal.rename(columns={"rate[U/hr]": "values"}, inplace=True)

        columns = ['time', 'values', 'delivery_type', 'duration[ms]']
        #df_insulin = pd.concat([df_bolus[columns], df_basal[columns]])
        df_insulin = df_bolus
        df_insulin = df_insulin.sort_values(by='time', ascending=True).reset_index(drop=True)
        def get_dose_type(x):
            if  x == 'temp':
                return DoseType.from_str("tempbasal")
            elif x == 'bolus':
                return DoseType.from_str("bolus")
            else:
                return DoseType.from_str("basal")

        dose_types = df_insulin.delivery_type.apply(
            lambda x: get_dose_type(x)).to_numpy()
        start_times = [timestamp for timestamp in df_insulin['time'].to_numpy()]
        end_times = df_insulin.apply(lambda x: x.time + pd.Timedelta(milliseconds=x['duration[ms]']), axis=1)
        end_times = [timestamp for timestamp in end_times]  # Strange error fix
        dose_values = df_insulin['values'].to_numpy()
        dose_delivered_units = [None for _ in range(len(dose_types))]

        return dose_types, start_times, end_times, dose_values, dose_delivered_units

    def get_carbohydrate_data(self, df_carbs):
        df_carbs = df_carbs.sort_values(by='time', ascending=True).reset_index(drop=True)

        carb_dates = [timestamp.to_pydatetime() for timestamp in df_carbs['time'].to_numpy()]
        carb_values = df_carbs['value'].to_numpy()
        carb_absorption_times = [value / 60 for value in df_carbs['absorption_time[s]'].to_numpy()]

        return carb_dates, carb_values, carb_absorption_times


    def get_input_dict(self):
        return ({
            'carb_value_units': 'g',
            'settings_dictionary':
                {
                    'model': [360.0, 75],
                    'momentum_data_interval': 15.0,
                    'suspend_threshold': None,
                    'dynamic_carb_absorption_enabled': True,
                    'retrospective_correction_integration_interval': 30,
                    'recency_interval': 15,
                    'retrospective_correction_grouping_interval': 30,
                    'rate_rounder': 0.05,
                    'insulin_delay': 10,
                    'carb_delay': 10,
                    'default_absorption_times': [120.0, 180.0, 240.0],
                    'max_basal_rate': 2.5,
                    'max_bolus': 12.0,
                    'retrospective_correction_enabled': True
                },
            'sensitivity_ratio_start_times': [datetime.time(0, 0), datetime.time(0, 30)],
            'sensitivity_ratio_end_times': [datetime.time(0, 30), datetime.time(0, 0)],
            'sensitivity_ratio_values': [86.4, 86.4],
            'sensitivity_ratio_value_units': 'mg/dL/U',

            'carb_ratio_start_times': [datetime.time(0, 0), datetime.time(8, 30)],
            'carb_ratio_values': [8.0, 8.0],
            'carb_ratio_value_units': 'g/U',

            'basal_rate_start_times': [datetime.time(0, 0), datetime.time(12, 0)],
            'basal_rate_minutes': [720.0, 720.0],
            'basal_rate_values': [0.7, 0.7],
            'basal_rate_units': 'U/hr',

            'target_range_start_times': [datetime.time(0, 0)],
            'target_range_end_times': [datetime.time(0, 0)],
            'target_range_minimum_values': [100.0],
            'target_range_maximum_values': [114.0],
            'target_range_value_units': 'mg/dL',

            'last_temporary_basal': []
        })

    def process_data(self, df, model_config_manager, real_time):
        # Implement library specific preprocessing steps that are required before training a pandas dataframe
        return df

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None




