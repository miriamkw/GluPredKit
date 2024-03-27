from glupredkit.models.base_model import BaseModel
from glupredkit.helpers.unit_config_manager import unit_config_manager
import datetime
import pandas as pd
import numpy as np

from pyloopkit.dose import DoseType
from pyloopkit.loop_math import predict_glucose
from pyloopkit.loop_data_manager import update


class Model(BaseModel):

    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.DIA = 360  # TODO: Is this the default in Loop?

    def fit(self, x_train, y_train):
        # Manually adjust the therapy settings at the bottom of this file to adjust the prediction model

        # TODO: Use a data-driven method to determine carb ratio, ISF and basal rate

        return self

    def predict(self, x_test):
        """
        Return:
        y_pred -- A list of lists of the predicted trajectories.
        """
        history_length = 24 * 12 + int(self.DIA / 5)  # = 360
        n_predictions = x_test.shape[0] - history_length

        if n_predictions <= 0:
            print("Not enough data to predict. Needs to be at least 24 hours plus duration of insulin absorption.")
            return

        # Note that the prediction output starts at the reference value, so element 1 is the first prediction
        prediction_index = int(self.prediction_horizon / 5)
        input_dict = self.get_input_dict()

        # For each glucose measurement, get a new prediction
        # Skip iteration if there are not enough predictions.
        y_pred = []

        # TODO: Remove this. Its only for the sake of testing
        n_predictions = 50

        print(f"Prediction number 0 of {n_predictions}")

        for i in range(0, n_predictions):
            df_subset = x_test.iloc[i:i + history_length]
            output_dict = self.get_prediction_output(df_subset, input_dict)

            if i % 50 == 0 and i != 0:  # Check if i is a multiple of 50 and not 0
                print(f"Prediction number {i} of {n_predictions}")

            if len(output_dict.get("predicted_glucose_values")) < prediction_index:
                print("Not enough predictions. Skipping iteration...")
                # TODO: Here we should just repeat the last predicted value until enough predictions
                continue
            else:
                current_predictions = output_dict.get("predicted_glucose_values")[1:prediction_index + 1]
                y_pred.append(current_predictions)

        return np.array(y_pred)

    def get_prediction_output(self, df, input_dict, time_to_calculate=None):
        # Important docs: https://github.com/miriamkw/PyLoopKit/blob/develop/pyloopkit/docs/pyloopkit_documentation.md
        # For correct predictions, at least 24 hours + duration of insulin absorption (DIA) of data is needed
        # NOTE: Not filtering away future glucose values will lead to erroneous prediction results!
        if not time_to_calculate:
            time_to_calculate = df.index[-1]

        history_length = 24 * 12 - int(self.DIA / 5)
        n_predictions = df.shape[0] - history_length

        if n_predictions <= 0:
            print("Not enough data to predict. Needs to be at least 24 hours plus duration of insulin absorption.")
            return

        filtered_df = df[df.index <= time_to_calculate].tail(history_length)

        input_dict["time_to_calculate_at"] = time_to_calculate

        input_dict["glucose_dates"] = filtered_df.index.tolist()
        input_dict["glucose_values"] = filtered_df.CGM.tolist()

        dose_types, start_times, end_times, dose_values, dose_delivered_units = self.get_insulin_data(df)

        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = start_times
        input_dict["dose_end_times"] = end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = dose_delivered_units

        input_dict["carb_dates"] = filtered_df[filtered_df.carbs > 0].index.tolist()
        input_dict["carb_values"] = filtered_df[filtered_df.carbs > 0].carbs.tolist()

        # Adding the default carb absorption time because it is not available in data sources.
        input_dict["carb_absorption_times"] = [180 for _ in filtered_df[filtered_df.carbs > 0].index.tolist()]

        return update(input_dict)


    def get_insulin_data(self, df):
        def get_dose_type(x):
            if x == 'temp':
                return DoseType.from_str("tempbasal")
            elif x == 'bolus':
                return DoseType.from_str("bolus")
            else:
                return DoseType.from_str("basal")

        # Using tempbasal as default, assuming that users are using
        basal_dose_types = [get_dose_type("tempbasal") for _ in df.index.tolist()]
        basal_start_times = df.index.tolist()
        basal_end_times = [val + pd.Timedelta(minutes=5) for val in df.index.tolist()]
        # TODO: Are basals in U or U/hr in dataframe? PyLoopKit is expecting U/hr
        # basal_values = [value / 5 * 60 for value in df.basal.tolist()]
        basal_values = df.basal.tolist()
        basal_units = [None for _ in df.index.tolist()]

        df_bolus = df.copy()[df.bolus > 0]
        bolus_dose_types = [get_dose_type("bolus") for _ in df_bolus.index.tolist()]
        bolus_start_times = df_bolus.index.tolist()
        bolus_end_times = df_bolus.index.tolist()
        bolus_values = df_bolus.bolus.tolist()
        bolus_units = [None for _ in df_bolus.index.tolist()]

        # Step 1: Combine into tuples
        combined_basal = list(zip(basal_dose_types, basal_start_times, basal_end_times, basal_values, basal_units))
        combined_bolus = list(zip(bolus_dose_types, bolus_start_times, bolus_end_times, bolus_values, bolus_units))

        # Step 2: Merge lists
        combined = combined_basal + combined_bolus

        # Step 3: Sort by the start times (second element of each tuple)
        combined.sort(key=lambda x: x[1])

        # Step 4: Separate into individual lists
        dose_types, start_times, end_times, values, units = zip(*combined)

        # Convert the tuples back to lists if needed
        dose_types = list(dose_types)
        start_times = list(start_times)
        end_times = list(end_times)
        values = list(values)
        units = list(units)

        return dose_types, start_times, end_times, values, units

    def get_input_dict(self):
        return ({
            'carb_value_units': 'g',
            'settings_dictionary':
                {
                    'model': [self.DIA, 75],
                    'momentum_data_interval': 15.0,
                    'suspend_threshold': None,
                    'dynamic_carb_absorption_enabled': True,
                    'retrospective_correction_integration_interval': 30,
                    'recency_interval': 15,
                    'retrospective_correction_grouping_interval': 30,
                    'rate_rounder': 0.05,
                    'insulin_delay': 10,
                    'carb_delay': 0,
                    'default_absorption_times': [120.0, 180.0, 240.0],
                    'max_basal_rate': 2.5,  # Doesn't matter for prediction, but mandatory to include
                    'max_bolus': 12.0,  # Doesn't matter for prediction, but mandatory to include
                    'retrospective_correction_enabled': True
                },
            'sensitivity_ratio_start_times': [datetime.time(0, 0)],
            'sensitivity_ratio_end_times': [datetime.time(0, 0)],
            'sensitivity_ratio_values': [45.0],
            'sensitivity_ratio_value_units': 'mg/dL/U',

            'carb_ratio_start_times': [datetime.time(0, 0)],
            'carb_ratio_values': [12.5],
            'carb_ratio_value_units': 'g/U',

            'basal_rate_start_times': [datetime.time(0, 0)],
            'basal_rate_minutes': [1440],  # the length of time the basal runs for (in minutes)
            'basal_rate_values': [0.9],  # the infusion rate in U/hour

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
