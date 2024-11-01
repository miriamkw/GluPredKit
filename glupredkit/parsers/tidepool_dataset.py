"""
The Tidepool Dataset parser is processing the data from the Researcher datasets and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import pandas as pd
import os
import numpy as np
from datetime import timedelta


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the T1DEXI dataset root folder.
        """
        # TODO: This of course has to be changed when we get the whole dataset
        HCL150_file_path = os.path.join(file_path, 'Tidepool-JDRF-HCL150-train', 'train-data')
        all_dfs = []
        all_ids = []

        for root, dirs, files in os.walk(HCL150_file_path):
            for file in files:
                if file.endswith('.csv'):
                    subject_file_path = os.path.join(root, file)
                    df = pd.read_csv(subject_file_path, low_memory=False)
                    all_dfs.append(df)

                    subject_id = file.split("_")[1].split(".")[0]
                    all_ids.append(subject_id)

        processed_dfs = []
        for index, df in enumerate(all_dfs):
            df_glucose, df_bolus, df_basal, df_carbs = self.get_dataframes(df)
            df_resampled = self.resample_data(df_glucose, df_bolus, df_basal, df_carbs)
            df_resampled['id'] = all_ids[index]
            processed_dfs.append(df_resampled)

        df_final = pd.concat(processed_dfs)

        # TODO: Update when you have all data
        df_final['is_test'] = False

        return df_final

    def resample_data(self, df_glucose, df_bolus, df_basal, df_carbs):
        df = df_glucose.copy()
        df = df['CGM'].resample('5T', label='right').mean()

        df_carbs = df_carbs.resample('5T', label='right').sum()
        df = pd.merge(df, df_carbs, on="date", how='outer')
        df['carbs'] = df['carbs']

        df_bolus = df_bolus.resample('5T', label='right').sum()
        df = pd.merge(df, df_bolus, on="date", how='outer')

        df_basal = df_basal.resample('5T', label='right').sum()
        df = pd.merge(df, df_basal, on="date", how='outer')

        df['insulin'] = df['bolus'] + df['basal'] / 12

        return df

    def get_dataframes(self, df):
        df['time'] = pd.to_datetime(df['time'])

        # Dataframe blood glucose
        # cbg = continuous blood glucose, smbg = self-monitoring of blood glucose
        df_glucose = df[df['type'] == 'cbg'][['time', 'units', 'uploadId', 'value']]
        df_glucose['value'] = df_glucose.apply(
            lambda row: row['value'] * 18.0182 if row['units'] == 'mmol/L' else row['value'], axis=1)
        df_glucose.rename(columns={"time": "date", "value": "CGM", 'uploadId': "id"}, inplace=True)
        df_glucose.drop(columns=['units'], inplace=True)
        df_glucose.sort_values(by='date', inplace=True, ascending=True)
        df_glucose.set_index('date', inplace=True)

        # Dataframe bolus doses
        df_bolus = df[df['type'] == 'bolus'][['time', 'normal']]
        df_bolus.rename(columns={"time": "date", "normal": "bolus"}, inplace=True)
        df_bolus.sort_values(by='date', inplace=True, ascending=True)
        df_bolus.set_index('date', inplace=True)

        # Dataframe basal rates
        df_basal = df[df['type'] == 'basal'][['time', 'deliveryType', 'duration', 'rate']]
        df_basal.rename(columns={"time": "date", "rate": "basal"}, inplace=True)
        df_basal['duration'] = df_basal['duration'] / 1000  # convert to seconds
        df_basal.sort_values(by='date', inplace=True, ascending=True)
        # We need to split each sample into five minute intervals, and create new rows for each five minutes
        expanded_rows = []
        for _, row in df_basal.iterrows():
            intervals = split_into_intervals(row)
            expanded_rows.extend(intervals)
        df_basal = pd.DataFrame(expanded_rows)
        df_basal.set_index('date', inplace=True)

        # Dataframe carbohydrates
        df_carbs = df[df['type'] == 'food'][['time', 'nutrition.carbohydrate.net']]
        df_carbs.rename(columns={"time": "date", "nutrition.carbohydrate.net": "carbs"}, inplace=True)
        df_carbs.sort_values(by='date', inplace=True, ascending=True)
        df_carbs.set_index('date', inplace=True)

        # TODO: Add workouts when available

        return df_glucose, df_bolus, df_basal, df_carbs


def split_into_intervals(row):
    start = row.date
    end = start + timedelta(seconds=row['duration'])
    total_insulin_delivered = row['basal'] * row['duration'] / 60
    interval = timedelta(minutes=5)

    # Track rows for the new DataFrame
    rows = []
    current = start

    while current < end:
        next_interval = min(current + interval, end)
        # Calculate the proportion of time in the 5-minute bin
        proportion = (next_interval - current) / timedelta(minutes=row['duration'])
        bin_value = total_insulin_delivered * proportion

        # Add the row to the list
        rows.append({
            'date': current,
            'basal': bin_value * 12  # Convert back to U/hr when we are done
        })

        # Move to the next 5-minute interval
        current = next_interval

    return rows


