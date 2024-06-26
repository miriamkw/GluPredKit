from glupredkit.models.base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
import ctypes
import pandas as pd
import json

class Model(BaseModel):

    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.basal = None
        self.insulin_sensitivity_factor = None
        self.carb_ratio = None

    def _fit_model(self, x_train, y_train, basal=0.75, insulin_sensitivity_factor=66.6, carb_ratio=9, *args):
        self.basal = basal
        self.insulin_sensitivity_factor = insulin_sensitivity_factor
        self.carb_ratio = carb_ratio

        return self

    def _predict_model(self, x_test):
        predictions = []
        n_predictions = self.prediction_horizon // 5

        # Load the dynamic Swift / C LoopAlgorithms library
        swift_lib = ctypes.CDLL('./libLoopAlgorithmToPython.dylib')

        # Specify the argument types and return type of the Swift function
        swift_lib.generatePrediction.argtypes = [ctypes.c_char_p]
        swift_lib.generatePrediction.restype = ctypes.POINTER(ctypes.c_double)

        count = 1
        for idx, row in x_test.iterrows():
            json_data = self.get_json_from_df(row, idx)
            json_bytes = json_data.encode('utf-8')  # Convert JSON string to bytes
            predictions += [self.get_predictions_from_json(swift_lib, json_bytes, n_predictions)]

            print(f'Prediction {count} of {x_test.shape[0]}')
            count += 1

        return predictions

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def get_json_from_df(self, row, start_test_date):

        # Dataframe blood glucose
        data = {"date": [start_test_date], "value": [row['CGM']]}
        df_glucose = pd.DataFrame(data)
        keep_running = True
        count = 1
        while keep_running:
            if 'CGM_' + str(count * 5) in row.index:
                new_date = start_test_date - pd.Timedelta(minutes=count * 5)
                new_value = row['CGM_' + str(count * 5)]
                new_row = pd.DataFrame({'date': [new_date], 'value': [new_value]})
                df_glucose = pd.concat([df_glucose, new_row], ignore_index=True)
                count += 1
            else:
                keep_running = False
        df_glucose['date'] = df_glucose['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_glucose['value'] = df_glucose['value'].astype(int)
        df_glucose.sort_values(by='date', inplace=True, ascending=True)

        # Dataframe carbohydrates
        if row['carbs'] > 0:
            data = {"date": [start_test_date], "grams": [row['carbs']]}
            df_carbs = pd.DataFrame(data)
        else:
            df_carbs = pd.DataFrame()
        keep_running = True
        count = 1
        while keep_running:
            keep_running = False
            if 'carbs_' + str(count * 5) in row.index:
                new_date = start_test_date - pd.Timedelta(minutes=count * 5)
                new_value = row['carbs_' + str(count * 5)]
                new_row = pd.DataFrame({'date': [new_date], 'grams': [new_value]})
                df_carbs = pd.concat([df_carbs, new_row], ignore_index=True)
                keep_running = True
            if 'carbs_what_if_' + str(count * 5) in row.index:
                new_date = start_test_date + pd.Timedelta(minutes=count * 5)
                new_value = row['carbs_what_if_' + str(count * 5)]
                new_row = pd.DataFrame({'date': [new_date], 'grams': [new_value]})
                df_carbs = pd.concat([df_carbs, new_row], ignore_index=True)
                keep_running = True
            if keep_running:
                count += 1

        # TODO: Add absorbtiontime in tidepool parser and here!
        df_carbs['absorptionTime'] = 10800
        df_carbs = df_carbs[df_carbs['grams'] > 0]
        df_carbs['date'] = df_carbs['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_carbs.sort_values(by='date', inplace=True, ascending=True)

        # Dataframe bolus doses
        if row['bolus'] > 0:
            data = {"startDate": [start_test_date], "endDate": [start_test_date], "volume": [row['bolus']], "type": "bolus"}
            df_bolus = pd.DataFrame(data)
        else:
            df_bolus = pd.DataFrame()
        keep_running = True
        count = 1
        while keep_running:
            keep_running = False
            if 'bolus_' + str(count * 5) in row.index:
                new_date = start_test_date - pd.Timedelta(minutes=count * 5)
                new_value = row['bolus_' + str(count * 5)]
                new_row = pd.DataFrame({'startDate': [new_date], 'endDate': [new_date], 'volume': [new_value],
                                        "type": "bolus"})
                df_bolus = pd.concat([df_bolus, new_row], ignore_index=True)
                keep_running = True
            if 'bolus_what_if_' + str(count * 5) in row.index:
                new_date = start_test_date + pd.Timedelta(minutes=count * 5)
                new_value = row['bolus_what_if_' + str(count * 5)]
                new_row = pd.DataFrame({'startDate': [new_date], 'endDate': [new_date], 'volume': [new_value],
                                        "type": "bolus"})
                df_bolus = pd.concat([df_bolus, new_row], ignore_index=True)
                keep_running = True
            if keep_running:
                count += 1
        df_bolus = df_bolus[df_bolus['volume'] > 0]
        df_bolus['startDate'] = df_bolus['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_bolus['endDate'] = df_bolus['endDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_bolus.sort_values(by='startDate', inplace=True, ascending=True)

        # Dataframe basal doses
        df_basal = pd.DataFrame()
        keep_running = True
        count = 1
        while keep_running:
            keep_running = False
            if 'basal_' + str(count * 5) in row.index:
                new_date = start_test_date - pd.Timedelta(minutes=count * 5)
                new_value = row['basal_' + str(count * 5)]
                new_row = pd.DataFrame({'startDate': [new_date], 'endDate': [new_date + pd.Timedelta(minutes=5)],
                                        'volume': [new_value], "type": "basal"})
                df_basal = pd.concat([df_basal, new_row], ignore_index=True)
                keep_running = True
            if 'basal_what_if_' + str(count * 5) in row.index:
                new_date = start_test_date + pd.Timedelta(minutes=count * 5)
                new_value = row['basal_what_if_' + str(count * 5)]
                new_row = pd.DataFrame({'startDate': [new_date], 'endDate': [new_date + pd.Timedelta(minutes=5)],
                                        'volume': [new_value], "type": "basal"})
                df_basal = pd.concat([df_basal, new_row], ignore_index=True)
                keep_running = True
            if keep_running:
                count += 1
        df_basal['startDate'] = df_basal['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_basal['endDate'] = df_basal['endDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_basal.sort_values(by='startDate', inplace=True, ascending=True)

        df_insulin = pd.concat([df_bolus, df_basal], ignore_index=True)
        df_insulin.sort_values(by='startDate', inplace=True, ascending=True)

        insulin_history_list = df_insulin.to_dict(orient='records')
        glucose_history_list = df_glucose.to_dict(orient='records')
        carbs_history_list = df_carbs.to_dict(orient='records')

        basal = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": self.basal}]
        carb_ratio = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z",
                       "value": self.carb_ratio}]
        ins_sens = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z",
                     "value": self.insulin_sensitivity_factor}]

        # Create the final JSON structure
        data = {
            "carbEntries": carbs_history_list,
            "doses": insulin_history_list,
            "glucoseHistory": glucose_history_list,
            "basal": basal,
            "carbRatio": carb_ratio,
            "sensitivity": ins_sens
        }
        return json.dumps(data, indent=2)

    def get_predictions_from_json(self, swift_lib, json_bytes, n):
        # Call the Swift function
        result = swift_lib.generatePrediction(json_bytes)
        glucose_array = [result[i] for i in range(1, n+1)]  # Read the array from the returned pointer
        """
        # Specify the argument types and return type of the Swift function
        swift_lib.getPredictionDates.argtypes = [ctypes.c_char_p]
        swift_lib.getPredictionDates.restype = ctypes.c_char_p

        # Call the Swift function
        result = swift_lib.getPredictionDates(json_bytes).decode('utf-8')
        date_list = result.split(',')[1:n+1]  # We drop the first element because it's redundant with the measured value
        """
        # TODO: Remember to validate that the predictions have 5 min intervals between them
        return glucose_array



