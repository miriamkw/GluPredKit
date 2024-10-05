from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from glupredkit.metrics.grmse import Metric
from loop_to_python_api.api import get_prediction_values_and_dates

import datetime
import numpy as np
import pandas as pd

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.subject_ids = None
        self.basal_rates = []
        self.insulin_sensitivites = []
        self.carb_ratios = []

    def _fit_model(self, x_train, y_train, n_cross_val_samples=200, *args):
        required_columns = ['CGM', 'carbs', 'basal', 'bolus']
        missing_columns = [col for col in required_columns if col not in x_train.columns]
        if missing_columns:
            raise ValueError(
                f"The Loop model requires the following features from the data input: {', '.join(missing_columns)}. "
                f"Please ensure that your dataset and configurations include these features. ")

        self.subject_ids = x_train['id'].unique()

        x_train['insulin'] = x_train['bolus'] + (x_train['basal'] / 12)

        grmse = Metric()

        for subject_id in self.subject_ids:

            x_train_filtered = x_train[x_train['id'] == subject_id]
            y_train_filtered = y_train[x_train['id'] == subject_id]

            subset_df_x = x_train_filtered.tail(n_cross_val_samples)
            subset_df_y = y_train_filtered.tail(n_cross_val_samples)

            # Flattened list of measured values across trajectory
            y_true = subset_df_y.to_numpy().ravel().tolist()

            """
            def iterative_search_rmse():
                # Generate multipliers from 0.5 to 1.5 in steps of 0.1
                multipliers = np.arange(0.5, 1.6, 0.1)

                # Store RMSE results
                rmse_results = {}
                rmse_values = []

                # Calculate RMSE for each multiplier, or reuse if already computed
                for m in multipliers:
                    print(f"\nIteration for multiplier: {m}")

                    y_true = subset_df_y
                    y_pred = self._predict_model(subset_df_x,
                                                 basal=m * daily_avg_insulin * 0.45 / 24,
                                                 isf=1800 / (m * daily_avg_insulin),
                                                 cr=500 / (m * daily_avg_insulin))

                    # Flatten lists
                    y_true = y_true.to_numpy().ravel().tolist()
                    if isinstance(y_pred, list) and any(isinstance(i, list) for i in y_pred):
                        y_pred = [item for sublist in y_pred for item in sublist]  # Flattening y_pred
                    else:
                        y_pred = y_pred  # Use as is if it's already flat

                    rmse = grmse(y_true, y_pred)
                    rmse_results[m] = rmse  # Cache the result
                    print(f"Result for {m}: {rmse}")

                    # Append the RMSE for this multiplier to the list
                    rmse_values.append(rmse_results[m])

                # Find the index of the minimum RMSE
                min_idx = np.argmin(rmse_values)
                best_multiplier = multipliers[min_idx]
                best_rmse = rmse_values[min_idx]

                # Return the best multiplier and the stored RMSE results
                print(f"Returning best multiplier: {best_multiplier}")
                return best_multiplier, rmse_results

            # Get the best TDD from binary search, and set the new therapy settings
            best_multiplier, rmse_results = iterative_search_rmse()
            print(f"Best Multiplier: {best_multiplier}")
            print(f"RMSE Results: {rmse_results}")

            basal = best_multiplier * daily_avg_insulin * 0.45 / 24  # Basal 45% of TDI
            isf = 1800 / (daily_avg_insulin * best_multiplier)
            carb_ratio = 500 / (daily_avg_insulin * best_multiplier)

            self.basal_rates += [basal]
            self.carb_ratios += [carb_ratio]
            self.insulin_sensitivites += [isf]
            """
            # Calculate total daily insulin
            daily_avg_insulin = np.mean(x_train_filtered.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))
            print(f"Daily average insulin for subject {subject_id}: ", daily_avg_insulin)

            daily_avg_basal = np.mean(subset_df_x.groupby(pd.Grouper(freq='D')).agg({'basal': 'mean'}))
            computed_basal = daily_avg_insulin * 0.45 / 24  # Basal 45% of TDI
            print(f"daily average basal is {daily_avg_basal}, while 45% of TDD is {computed_basal}")

            basal = (daily_avg_basal + computed_basal) / 2  # Average between 45% and their original setting
            self.basal_rates += [basal]  # Basal average of the daily basal rate of the person
            print(f"Basal for subject {subject_id}: ", basal)

            isf = 1800 / daily_avg_insulin  # ISF 1800 rule
            cr = 500 / daily_avg_insulin  # CR 500 rule

            """
            mult_factors = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

            best_grmse = np.inf
            best_isf = isf
            best_cr = cr
            for i in mult_factors:
                for j in mult_factors:
                    y_pred = self._predict_model(subset_df_x, basal=basal, isf=isf*i, cr=cr*j)

                    # Flatten y_pred
                    if isinstance(y_pred, list) and any(isinstance(i, list) for i in y_pred):
                        y_pred = [item for sublist in y_pred for item in sublist]  # Flattening y_pred
                    else:
                        y_pred = y_pred  # Use as is if it's already flat

                    print(f'Factors {i} and {j}')

                    iteration_result = grmse(y_true, y_pred)
                    print("gRMSE: ", iteration_result)
                    # print("Distribution difference: ", int(np.std(last_y_pred) - std_target))
                    if iteration_result < best_grmse:
                        best_isf = isf*i
                        best_cr = cr*j
                        best_grmse = iteration_result

            self.insulin_sensitivites += [best_isf]
            self.carb_ratios += [best_cr]
            """
            self.insulin_sensitivites += [isf]
            self.carb_ratios += [cr]

        return self

    def _predict_model(self, x_test, basal=None, isf=None, cr=None):
        n_predictions = self.prediction_horizon // 5
        y_pred = []

        for subject_idx, subject_id in enumerate(self.subject_ids):
            df_subset = x_test[x_test['id'] == subject_id]

            for _, row in df_subset.iterrows():
                json_data = self.get_json_from_df(row, subject_idx, basal=basal, isf=isf, cr=cr)
                predictions, dates = get_prediction_values_and_dates(json_data)

                # Skipping first predicted sample because it is repeating the reference value
                y_pred += [predictions[1:n_predictions + 1]]

        return y_pred

    def best_params(self):
        best_params = [{
            "basal rates": self.basal_rates,
            "insulin sensitivities": self.insulin_sensitivites,
            "carbohydrate ratios": self.carb_ratios,
        }]

        return best_params

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)


    def get_json_from_df(self, data, id_index, basal=None, isf=None, cr=None):

        def get_dates_and_values(column, data):
            relevant_columns = [val for val in data.index if val.startswith(column)]
            dates = []
            values = []

            date = data.name
            if isinstance(date, np.int64):
                date = datetime.datetime.now()

            for col in relevant_columns:
                if col == column:
                    values.append(data[col])
                    dates.append(date)
                elif "what_if" in col:
                    values.append(data[col])
                    new_date = date + datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)
                else:
                    values.append(data[col])
                    new_date = date - datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)

            if not dates or not values:
                # Handle the case where one or both lists are empty
                pass
            else:
                # Sorting
                combined = list(zip(dates, values))
                # Sort by the dates (first element of each tuple)
                combined.sort(key=lambda x: x[0])
                # Separate into individual lists
                dates, values = zip(*combined)

            return dates, values

        bolus_dates, bolus_values = get_dates_and_values('bolus', data)
        basal_dates, basal_values = get_dates_and_values('basal', data)

        insulin_json_list = []
        for date, value in zip(bolus_dates, bolus_values):
            entry = {
                "startDate": date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "endDate": (date + datetime.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "type": 'bolus',
                "volume": value
            }
            insulin_json_list.append(entry)

        for date, value in zip(basal_dates, basal_values):
            entry = {
                "startDate": date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "endDate": (date + datetime.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "type": 'basal',
                "volume": value / 12 # Converting from U/hr to delivered units in 5 minutes
            }
            insulin_json_list.append(entry)
        insulin_json_list.sort(key=lambda x: x['startDate'])

        bg_dates, bg_values = get_dates_and_values('CGM', data)
        bg_json_list = []
        for date, value in zip(bg_dates, bg_values):
            entry = {
                "date": date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "value": value
            }
            bg_json_list.append(entry)
        bg_json_list.sort(key=lambda x: x['date'])

        carbs_dates, carbs_values = get_dates_and_values('carbs', data)
        carbs_json_list = []
        for date, value in zip(carbs_dates, carbs_values):
            entry = {
                "date": date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "grams": value,
                "absorptionTime": 10800,
            }
            carbs_json_list.append(entry)
        carbs_json_list.sort(key=lambda x: x['date'])

        # It is important that the setting dates encompass the data to avoid a code crash
        start_date_settings = datetime.datetime.fromisoformat(bg_json_list[0]['date'].replace('Z', '+00:00')) - datetime.timedelta(hours=999)
        end_date_settings = datetime.datetime.fromisoformat(bg_json_list[-1]['date'].replace('Z', '+00:00')) + datetime.timedelta(hours=999)

        start_date_str = start_date_settings.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date_settings.strftime('%Y-%m-%dT%H:%M:%SZ')

        basal = [{
            "startDate": start_date_str,
            "endDate": end_date_str,
            "value": basal if basal is not None else self.basal_rates[id_index]
        }]

        isf = [{
            "startDate": start_date_str,
            "endDate": end_date_str,
            "value": isf if isf is not None else self.insulin_sensitivites[id_index]
        }]

        cr = [{
            "startDate": start_date_str,
            "endDate": end_date_str,
            "value": cr if cr is not None else self.carb_ratios[id_index]
        }]

        json_data = {
            "carbEntries": carbs_json_list,
            "doses": insulin_json_list,
            "glucoseHistory": bg_json_list,
            "basal": basal,
            "carbRatio": cr,
            "sensitivity": isf,
        }
        return json_data




