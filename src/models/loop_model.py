import numpy as np

from src.models.base_model import BaseModel
from typing import List
import datetime
from pyloopkit.loop_data_manager import update
from pyloopkit.dose import DoseType
import pandas as pd
import pytz

class LoopModel(BaseModel):
    def __init__(self, output_offsets: List[int] = None):
        if output_offsets is None:
            output_offsets = list(range(5, 365, 5))  # default offsets
        self.output_offsets = output_offsets

    def fit(self, df_glucose, df_bolus, df_basal, df_carbs):
        # Manually adjust the therapy settings at the bottom of this file to adjust the prediction model
        return self

    def predict(self, df_glucose, df_bolus, df_basal, df_carbs):
        # TODO: Accounting for time zones in the predictions
        # TODO: Verify that predicted and measured values are in the same time grid (we have assumed a new measurement every 5 minutes)

        input_dict = self.get_input_dict()

        dose_types, start_times, end_times, dose_values, dose_delivered_units = self.get_insulin_data(df_bolus, df_basal)
        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = start_times
        input_dict["dose_end_times"] = end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = dose_delivered_units

        carb_dates, carb_values, carb_absorption_times = self.get_carbohydrate_data(df_carbs)
        input_dict["carb_dates"] = carb_dates
        input_dict["carb_values"] = carb_values
        input_dict["carb_absorption_times"] = carb_absorption_times

        history_length = 6
        n_predictions = df_glucose.shape[0] - 12*6 - history_length

        if n_predictions <= 0:
            print("Not enough data to predict.")
            return

        # Sort glucose values
        df_glucose = df_glucose.sort_values(by='time', ascending=True).reset_index(drop=True)

        # For each glucose measurement, get a new prediction
        # Skip iteration if there are not enough predictions.
        y_pred = []
        y_true = []
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
                y_pred.append(output_dict.get("predicted_glucose_values")[1:73])
                if df_glucose['units'][0] == 'mmol/L':
                    y_true.append([value * 18.0182 for value in df_glucose['value'][i:i+72]])
                else:
                    y_true.append(df_glucose['value'][i:i+72])

        return y_pred, y_true

    def get_glucose_data(self, df):
        # Sort glucose values
        #df_glucose = df_glucose.sort_values(by='time', ascending=True).reset_index(drop=True)
        glucose_dates = [timestamp for timestamp in df['time'].to_numpy()]
        if df['units'][0] == 'mmol/L':
            glucose_values = [value * 18.0182 for value in df['value'].to_numpy()]
        else:
            glucose_values = df['value'].to_numpy()
        return glucose_dates, glucose_values

    def get_insulin_data(self, df_bolus, df_basal):
        # Is sorting is required?
        df_bolus = df_bolus.sort_values(by='time', ascending=True).reset_index(drop=True)
        df_basal = df_basal.sort_values(by='time', ascending=True).reset_index(drop=True)

        # TODO: Insulin doses need to be sorted after merge of bolus doses and basal rates

        dose_types_bolus = [DoseType.from_str("bolus") for _ in df_bolus['dose[IU]']]
        dose_types_basal = df_basal.delivery_type.apply(
            lambda x: DoseType.from_str("tempbasal") if x == 'temp' else DoseType.from_str("basal")).to_numpy()
        dose_types = np.concatenate((dose_types_bolus, dose_types_basal), axis=0)

        start_times_bolus = [timestamp for timestamp in df_bolus['time'].to_numpy()]
        start_times_basal = [timestamp for timestamp in df_basal['time'].to_numpy()]
        start_times = np.concatenate((start_times_bolus, start_times_basal), axis=0)
        start_times = [timestamp for timestamp in start_times]  # Strange error fix

        end_times_bolus = start_times_bolus
        end_times_basal = df_basal.apply(lambda x: x.time + pd.Timedelta(milliseconds=x['duration[ms]']),
                                         axis=1)  # .to_numpy()
        end_times_basal = [timestamp for timestamp in end_times_basal.to_numpy()]
        end_times = np.concatenate((end_times_bolus, end_times_basal), axis=0)
        end_times = [timestamp for timestamp in end_times]  # Strange error fix

        dose_values_bolus = df_bolus['dose[IU]'].to_numpy()
        dose_values_basal = df_basal['rate[IU]'].to_numpy()
        dose_values = np.concatenate((dose_values_bolus, dose_values_basal), axis=0)

        dose_delivered_units = [None for i in range(len(dose_types))]

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
                    'retrospective_correction_enabled': None
                },
            'sensitivity_ratio_start_times': [datetime.time(0, 0), datetime.time(0, 30)],
            'sensitivity_ratio_end_times': [datetime.time(0, 30), datetime.time(0, 0)],
            'sensitivity_ratio_values': [81.0, 81.0],
            'sensitivity_ratio_value_units': 'mg/dL/U',

            'carb_ratio_start_times': [datetime.time(0, 0), datetime.time(8, 30)],
            'carb_ratio_values': [10.0, 10.0],
            'carb_ratio_value_units': 'g/U',

            'basal_rate_start_times': [datetime.time(0, 0), datetime.time(12, 0)],
            'basal_rate_minutes': [720.0, 720.0],
            'basal_rate_values': [0.8, 0.8],
            'basal_rate_units': 'U/hr',

            'target_range_start_times': [datetime.time(0, 0)],
            'target_range_end_times': [datetime.time(0, 0)],
            'target_range_minimum_values': [100.0],
            'target_range_maximum_values': [114.0],
            'target_range_value_units': 'mg/dL',

            'last_temporary_basal': []
        })
