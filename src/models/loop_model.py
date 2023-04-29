from src.models.base_model import BaseModel
from typing import List
import datetime
from pyloopkit.loop_data_manager import update

class LoopModel(BaseModel):
    def __init__(self, output_offsets: List[int] = None):
        if output_offsets is None:
            output_offsets = list(range(5, 365, 5))  # default offsets
        self.output_offsets = output_offsets

    def fit(self, df_glucose, df_bolus, df_basal, df_carbs):
        # Manually adjust the therapy settings at the bottom of this file to adjust the prediction model
        return self

    def predict(self, df_glucose, df_bolus, df_basal, df_carbs):
        input_dict = self.get_input_dict()
        df_glucose = df_glucose.sort_values(by='time', ascending=True)

        input_dict["glucose_dates"] = [timestamp.to_pydatetime() for timestamp in df_glucose['time'].to_numpy()]
        if df_glucose['units'][0] == 'mmol/L':
            input_dict["glucose_values"] = [value*18.0182 for value in df_glucose['value'].to_numpy()]
        else:
            input_dict["glucose_values"] = df_glucose['value'].to_numpy()

        input_dict["dose_types"] = []
        input_dict["dose_start_times"] = []
        input_dict["dose_end_times"] = []
        input_dict["dose_values"] = []

        input_dict["carb_dates"] = []
        input_dict["carb_values"] = []
        input_dict["carb_absorption_times"] = []

        # For each glucose measurement in data, get a new prediction (input_dict['time_to_calculate_at': <SOME DATE>])
        # Skip iteration if there is not enough history or future measurements
        # For now we do it simple: First measurement after 12 measurements, and last when we have 12*6 more measurements left

        input_dict["time_to_calculate_at"] = input_dict["glucose_dates"][20]
        output_dict = update(input_dict)

        """
        QUESTIONS:
        Changing "time_to_calculate_at" makes the predictions different, but not the dates. I dont understand what happens.
        Update expects a pretty specific input: we have to sort out too old AND too new values, if not, I do not 
        understand the output.
        
        Should I just implement and conform to this, even though it will lead to a slow runtime?
        """

        print("OUTPUT")
        print(output_dict.get("predicted_glucose_values"))
        print(output_dict.get("predicted_glucose_dates"))
        print("Prediction date: ", input_dict["time_to_calculate_at"])

        y_pred, y_test = [], []

        return y_pred, y_test

    def process_data(self, df_glucose, df_bolus, df_basal, df_carbs):
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
            'dose_delivered_units': [],

            'target_range_start_times': [datetime.time(0, 0)],
            'target_range_end_times': [datetime.time(0, 0)],
            'target_range_minimum_values': [100.0],
            'target_range_maximum_values': [114.0],
            'target_range_value_units': 'mg/dL',

            'last_temporary_basal': []
        })
