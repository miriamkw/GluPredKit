from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
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

    def _fit_model(self, x_train, y_train, n_cross_val_samples=1000, *args):
        required_columns = ['CGM', 'carbs', 'basal', 'bolus']
        missing_columns = [col for col in required_columns if col not in x_train.columns]
        if missing_columns:
            raise ValueError(
                f"The Loop model requires the following features from the data input: {', '.join(missing_columns)}. "
                f"Please ensure that your dataset and configurations include these features. ")

        self.subject_ids = x_train['id'].unique()

        x_train['insulin'] = x_train['bolus'] + (x_train['basal'] / 12)

        for subject_id in self.subject_ids:

            x_train_filtered = x_train[x_train['id'] == subject_id]
            y_train_filtered = y_train[x_train['id'] == subject_id]

            subset_df_x = x_train_filtered.tail(n_cross_val_samples)
            subset_df_y = y_train_filtered.tail(n_cross_val_samples)

            daily_avg_basal = np.mean(subset_df_x.groupby(pd.Grouper(freq='D')).agg({'basal': 'mean'}))
            print("daily average basal: ", daily_avg_basal)

            # Calculate total daily insulin
            daily_avg_insulin = np.mean(x_train_filtered.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))
            print(f"Daily average insulin for subject {subject_id}: ", daily_avg_insulin)

            basal = daily_avg_insulin * 0.45 / 24  # Basal 45% of TDI
            print("Basal 45%: ", basal)

            isf = 1800 / daily_avg_insulin
            print("1800 rule isf: ", isf)

            carb_ratio = 500 / daily_avg_insulin
            print("500 rule: ", carb_ratio)

            self.basal_rates += [basal]
            self.carb_ratios += [carb_ratio]
            self.insulin_sensitivites += [isf]

            # TODO: Tune the therapy settings (create linear combination of basal, cr and isf by multiplying a factor to daily avg insulin and use binary search)
            # With that, we just create one parameter to decide the "agressiveness" of the algorithm
            # We use that to minimize the grmse
            #self._predict_model(subset_df_x)

        return self

    def _predict_model(self, x_test):
        n_predictions = self.prediction_horizon // 5
        y_pred = []

        for i in range(len(self.subject_ids)):
            for _, row in x_test.iterrows():
                json_data = self.get_json_from_df(row, i)
                predictions, dates = get_prediction_values_and_dates(json_data)

                # Skipping first predicted sample because it is repeating the reference value
                y_pred += [predictions[1:n_predictions + 1]]

        print(y_pred)
        print(len(y_pred))
        print(len(y_pred[0]))

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


    def get_json_from_df(self, data, id_index):

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

        #print("DATES", dates)
        #print("VALUES", values)
        #print("DATA", data[[col for col in data.index if col.startswith('basal')]])

        # It is important that the setting dates encompass the data to avoid a code crash
        start_date_settings = datetime.datetime.fromisoformat(bg_json_list[0]['date'].replace('Z', '+00:00')) - datetime.timedelta(hours=999)
        end_date_settings = datetime.datetime.fromisoformat(bg_json_list[-1]['date'].replace('Z', '+00:00')) + datetime.timedelta(hours=999)

        start_date_str = start_date_settings.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date_settings.strftime('%Y-%m-%dT%H:%M:%SZ')

        basal = [{
            "startDate": start_date_str,
            "endDate": end_date_str,
            "value": self.basal_rates[id_index]
        }]

        isf = [{
            "startDate": start_date_str,
            "endDate": end_date_str,
            "value": self.insulin_sensitivites[id_index]
        }]

        cr = [{
            "startDate": start_date_str,
            "endDate": end_date_str,
            "value": self.carb_ratios[id_index]
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


















