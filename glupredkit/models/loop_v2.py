from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from loop_to_python_api.api import get_prediction_values_and_dates
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
            print(subject_id)
            # TODO: Tune the therapy settings

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
            print("Basal 45\%: ", basal)

            isf = 1800 / daily_avg_insulin
            print("1800 rule isf: ", isf)

            carb_ratio = 500 / daily_avg_insulin
            print("500 rule: ", carb_ratio)

            self._predict_model(x_train)



        return self

    def _predict_model(self, x_test):
        y_pred = []

        predictions = get_prediction_values_and_dates(x_test)


        return y_pred

    def best_params(self):
        best_params = []

        return best_params

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)


