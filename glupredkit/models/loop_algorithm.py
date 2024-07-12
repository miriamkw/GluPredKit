from glupredkit.models.base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from skopt import gp_minimize
from skopt.space import Real
from glupredkit.metrics.mcc_hypo import Metric as MCC_Hypo
from glupredkit.metrics.mcc_hyper import Metric as MCC_Hyper
import ctypes
import pandas as pd
import numpy as np
import json

class Model(BaseModel):

    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.basal = None
        self.insulin_sensitivity_factor = None
        self.carb_ratio = None

    def _fit_model(self, x_train, y_train, n_cross_val_samples=1000, tuning=True, *args):
        # TODO: Train per id, change therapy settings to lists
        self.basal = 0.75
        self.insulin_sensitivity_factor = 66.6
        self.carb_ratio = 9

        if tuning:
            sampled_indices = x_train.sample(n=n_cross_val_samples, random_state=42).index
            subset_df_x = x_train.loc[sampled_indices]
            subset_df_y = y_train.loc[sampled_indices]

            # Calculate total daily insulin
            daily_avg_insulin = np.mean(x_train.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))
            print(f"Daily average insulin: ", daily_avg_insulin)

            initial_basal = daily_avg_insulin * 0.5 / 24  # Basal 50% of TDI
            initial_isf = 1800 / daily_avg_insulin  # ISF 1800 rule
            initial_cr = 500 / daily_avg_insulin  # CR 500 rule

            lower_multiplication_factor = 0.5
            upper_multiplication_factor = 2.0

            # Use the model parameters to predict sequences
            def custom_model(params):
                self.basal = params[0]
                self.insulin_sensitivity_factor = params[1]
                self.carb_ratio = params[2]

                y_pred = self._predict_model(subset_df_x)
                flattened_predictions = [item for sublist in y_pred for item in sublist]
                return flattened_predictions

            # Convert continuous values to categories
            def categorize(values, a, b):
                categories = np.digitize(values, bins=[a, b], right=True)
                return categories

            # True categories for the target sequence
            low_threshold = 80
            high_threshold = 180

            true_values = subset_df_y.values.tolist()
            flattened_true_values = [item for sublist in true_values for item in sublist]
            true_categories = np.array(categorize(flattened_true_values, a=low_threshold, b=high_threshold))

            # Using inverse frequency, assigning higher weights to less frequent categories
            num_categories = 3
            category_counts = np.array([np.sum(true_categories == i) for i in range(num_categories)])
            weights = 1 / category_counts
            weights = weights / np.sum(weights)  # Normalize to ensure the weights sum to 1
            print("Weights: ", weights)

            # Define your objective function
            def objective(params):
                # Get the sequence from your model
                sequence = custom_model(params)

                # Categorize the predicted sequence
                predicted_categories = categorize(sequence, a=low_threshold, b=high_threshold)

                squared_errors = [(true_categories == i) & (predicted_categories != i) for i in range(num_categories)]
                squared_errors = [np.sum(errors) / len(true_categories) for errors in squared_errors]
                weighted_squared_errors = [error * weight for error, weight in zip(squared_errors, weights)]
                total_error = np.sum(weighted_squared_errors)

                print(f'Params: {params}, MSE: {total_error * 10000}')

                return total_error

            def objective_mcc(params):
                mcc_hypo = MCC_Hypo()
                mcc_hyper = MCC_Hyper()

                flattened_predictions = custom_model(params)

                hypo_results = mcc_hypo(flattened_true_values, flattened_predictions)
                hyper_results = mcc_hyper(flattened_true_values, flattened_predictions)
                total_error = np.mean([hypo_results, hyper_results])

                print(f'Hypo results: {hypo_results}, hyper results: {hyper_results}, mean: {total_error}')

                # Add 1 and invert to make so that lower is better
                total_error = 2 - (total_error + 1)

                print(f'Params: {params}, MCC: {total_error}')

                return total_error

            def objective_mse(params):
                flattened_predictions = custom_model(params)
                res = np.square(np.subtract(flattened_true_values, flattened_predictions)).mean()

                print(f'MSE: {res}, RMSE: {np.sqrt(res)}')
                print(f'Params: {params}')

                return res

            # Define the parameter space
            param_space = [
                Real(initial_basal * lower_multiplication_factor, initial_basal * upper_multiplication_factor, name='basal'),
                Real(initial_isf * lower_multiplication_factor, initial_isf * upper_multiplication_factor, name='isf'),
                Real(initial_cr * lower_multiplication_factor, initial_cr * upper_multiplication_factor, name='cr')
            ]

            # Run Bayesian Optimization. We treat
            result = gp_minimize(
                objective_mse,  # Objective function
                param_space,  # Parameter space
                n_calls=100,  # Number of evaluations
                random_state=0  # For reproducibility
            )

            # Print the results
            print("Best score achieved: ", result.fun)
            print("Best parameters: ", result.x)

            self.basal = result.x[0]
            self.insulin_sensitivity_factor = result.x[1]
            self.carb_ratio = result.x[2]

        print(f"Therapy settings: ISF {self.insulin_sensitivity_factor}, CR: {self.carb_ratio}, basal: {self.basal}")

        return self

    def _predict_model(self, x_test):
        predictions = []
        n_predictions = self.prediction_horizon // 5

        count = 1
        print("Starting predicting...")
        for idx, row in x_test.iterrows():
            json_data = self.get_json_from_df(row, idx)
            json_bytes = json_data.encode('utf-8')  # Convert JSON string to bytes
            predictions += [self.get_predictions_from_json(json_bytes, n_predictions)]

            if count % 1000 == 0:
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

    def get_predictions_from_json(self, json_bytes, n):
        # Load the dynamic Swift / C LoopAlgorithms library
        swift_lib = ctypes.CDLL('./libLoopAlgorithmToPython.dylib')

        # Specify the argument types and return type of the Swift function
        swift_lib.generatePrediction.argtypes = [ctypes.c_char_p]
        swift_lib.generatePrediction.restype = ctypes.POINTER(ctypes.c_double)

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



