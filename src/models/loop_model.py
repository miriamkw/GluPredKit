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

        # Sort glucose values
        df_glucose = df_glucose.sort_values(by='time', ascending=True).reset_index(drop=True)

        time_to_calculate = df_glucose["time"][6]
        input_dict["time_to_calculate_at"] = time_to_calculate

        # NOTE: Not filtering away future glucose values will lead to erroneous prediction results!
        df_glucose = df_glucose[df_glucose['time'] < time_to_calculate]
        input_dict["glucose_dates"] = [timestamp for timestamp in df_glucose['time'].to_numpy()]
        if df_glucose['units'][0] == 'mmol/L':
            input_dict["glucose_values"] = [value*18.0182 for value in df_glucose['value'].to_numpy()]
        else:
            input_dict["glucose_values"] = df_glucose['value'].to_numpy()

        dose_types_bolus = [DoseType.from_str("bolus") for _ in df_bolus['dose[IU]']]
        dose_types_basal = df_basal.delivery_type.apply(lambda x: DoseType.from_str("tempbasal") if x == 'temp' else DoseType.from_str("basal")).to_numpy()
        dose_types = np.concatenate((dose_types_bolus, dose_types_basal), axis=0)

        start_times_bolus = [timestamp for timestamp in df_bolus['time'].to_numpy()]
        start_times_basal = [timestamp for timestamp in df_basal['time'].to_numpy()]
        start_times = np.concatenate((start_times_bolus, start_times_basal), axis=0)
        start_times = [timestamp for timestamp in start_times] # Strange error fix

        end_times_bolus = start_times_bolus
        end_times_basal = df_basal.apply(lambda x: x.time + pd.Timedelta(milliseconds=x['duration[ms]']), axis=1)#.to_numpy()
        end_times_basal = [timestamp for timestamp in end_times_basal.to_numpy()]
        end_times = np.concatenate((end_times_bolus, end_times_basal), axis=0)
        end_times = [timestamp for timestamp in end_times] # Strange error fix

        dose_values_bolus = df_bolus['dose[IU]'].to_numpy()
        dose_values_basal = df_basal['rate[IU]'].to_numpy()
        dose_values = np.concatenate((dose_values_bolus, dose_values_basal), axis=0)

        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = start_times
        input_dict["dose_end_times"] = end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = [None for i in range(len(dose_types))]

        # Is sorting is required?
        df_carbs = df_carbs.sort_values(by='time', ascending=True).reset_index(drop=True)
        input_dict["carb_dates"] = [timestamp.to_pydatetime() for timestamp in df_carbs['time'].to_numpy()]
        input_dict["carb_values"] = df_carbs['value'].to_numpy()
        input_dict["carb_absorption_times"] = [value/60 for value in df_carbs['absorption_time[s]'].to_numpy()]

        # For each glucose measurement in data, get a new prediction (input_dict['time_to_calculate_at': <SOME DATE>])
        # Skip iteration if there is not enough predictions
        # For now we do it simple: First prediction after 6 measurements, and last when we have 12*6 more measurements left

        output_dict = update(input_dict)

        """
        Changing "time_to_calculate_at" changes the prediction value output, but not the predictions dates output. 
        I dont understand what exactly this input does. The Update() function from pyloopkit expects to get 
        input data where too old and too new values are filtered out. 
        """

        print("OUTPUT")
        # NOTE: The output of the predictions includes the last measured glucose value
        print("N predictions: ", len(output_dict.get("predicted_glucose_values")))
        print("N dates: ", len(output_dict.get("predicted_glucose_dates")))
        print(output_dict.get("predicted_glucose_values")[:10])
        print(output_dict.get("predicted_glucose_dates")[:10])
        print("Prediction date: ", input_dict["time_to_calculate_at"])
        print("Reference glucose value: ", input_dict["glucose_values"][5])
        print("Reference glucose date: ", input_dict["glucose_dates"][5])

        y_pred, y_test = [], []

        return y_pred, y_test

    def get_glucose_data(self, df_glucose):
        return {}





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
