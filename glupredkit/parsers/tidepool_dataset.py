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
        file_path -- the file path to the tidepool dataset root folder.
        """
        file_paths = {
            'HCL150': ['Tidepool-JDRF-HCL150-train', 'Tidepool-JDRF-HCL150-test'],
            'SAP100': ['Tidepool-JDRF-SAP100-train', 'Tidepool-JDRF-SAP100-test'],
            'PA50': ['Tidepool-JDRF-PA50-train', 'Tidepool-JDRF-PA50-test']
        }
        all_dfs, all_ids, is_test_bools = [], [], []
        for prefix, folders in file_paths.items():
            for folder in folders:
                current_file_path = os.path.join(file_path, folder, 'train-data' if 'train' in folder else 'test-data')
                is_test = True if 'test' in folder else False
                all_dfs, all_ids, is_test_bools = get_dfs_and_ids(current_file_path, all_dfs, all_ids, is_test_bools, is_test, id_prefix=f'{prefix}-')

        processed_dfs = []
        for index, df in enumerate(all_dfs):
            df_glucose, df_bolus, df_basal, df_carbs, df_workouts = self.get_dataframes(df)
            df_resampled = self.resample_data(df_glucose, df_bolus, df_basal, df_carbs, df_workouts)
            df_resampled['id'] = all_ids[index]
            df_resampled['is_test'] = is_test_bools[index]

            processed_dfs.append(df_resampled)

        df_final = pd.concat(processed_dfs)
        return df_final

    def resample_data(self, df_glucose, df_bolus, df_basal, df_carbs, df_workouts):
        df = df_glucose.copy()
        df = df['CGM'].resample('5T', label='right').mean()

        df_carbs = df_carbs.resample('5T', label='right').sum()
        df = pd.merge(df, df_carbs, on="date", how='outer')

        df_bolus = df_bolus.resample('5T', label='right').sum()
        df = pd.merge(df, df_bolus, on="date", how='outer')

        df_basal = df_basal.resample('5T', label='right').sum()
        df = pd.merge(df, df_basal, on="date", how='outer')

        if not df_workouts.empty:
            df_workout_labels = df_workouts['workout_label'].resample('5T', label='right').last()
            df_calories_burned = df_workouts['calories_burned'].resample('5T', label='right').sum()
            df = pd.merge(df, df_workout_labels, on="date", how='outer')
            df = pd.merge(df, df_calories_burned, on="date", how='outer')

        df['insulin'] = df['bolus'] + df['basal'] / 12
        return df

    def get_dataframes(self, df):
        df['time'] = pd.to_datetime(df['time'])

        # TODO: For time zone local, there are for a subset a column "est.localTime" that we could use

        # Dataframe blood glucose
        # cbg = continuous blood glucose, smbg = self-monitoring of blood glucose
        #df_glucose = df[df['type'] == 'cbg'][['time', 'units', 'value']]
        df_glucose = df[df['type'].isin(['cbg', 'smbg'])][['time', 'units', 'value']]
        df_glucose['value'] = df_glucose.apply(
            lambda row: row['value'] * 18.0182 if row['units'] == 'mmol/L' else row['value'], axis=1)
        df_glucose.rename(columns={"time": "date", "value": "CGM"}, inplace=True)
        df_glucose.drop(columns=['units'], inplace=True)
        df_glucose.sort_values(by='date', inplace=True, ascending=True)
        df_glucose.set_index('date', inplace=True)

        # Dataframe bolus doses
        df_bolus = df[df['type'] == 'bolus'][['time', 'normal']]
        df_bolus.rename(columns={"time": "date", "normal": "bolus"}, inplace=True)
        df_bolus.sort_values(by='date', inplace=True, ascending=True)
        df_bolus.set_index('date', inplace=True)

        # Dataframe basal rates
        # TODO: Verify that basals are correctly added according to schedule / percentage / temp basals...
        df_basal = df[df['type'] == 'basal'][['time', 'duration', 'rate', 'units']]
        df_basal.rename(columns={"time": "date", "rate": "basal"}, inplace=True)
        df_basal['duration'] = df_basal['duration'] / 1000  # convert to seconds
        df_basal.sort_values(by='date', inplace=True, ascending=True)
        # We need to split each sample into five minute intervals, and create new rows for each five minutes
        expanded_rows = []
        for _, row in df_basal.iterrows():
            intervals = split_basal_into_intervals(row)
            expanded_rows.extend(intervals)
        df_basal = pd.DataFrame(expanded_rows)
        df_basal.set_index('date', inplace=True)

        # Dataframe carbohydrates
        df_carbs = pd.DataFrame()
        if 'nutrition.carbohydrate.net' in df.columns:
            df_carbs = df[df['type'] == 'food'][['time', 'nutrition.carbohydrate.net']]
            df_carbs.rename(columns={"time": "date", "nutrition.carbohydrate.net": "carbs"}, inplace=True)
        elif 'carbInput' in df.columns:
            df_carbs = df[['time', 'carbInput', 'type']][df['carbInput'].notna()]
            df_carbs.rename(columns={"time": "date", "carbInput": "carbs"}, inplace=True)
        else:
            for col in df.columns:
                print(f"{col}: {df[col].unique()[:8]}")
            assert ValueError("No carbohydrate data detected for subject! Inspect data.")
        df_carbs.sort_values(by='date', inplace=True, ascending=True)
        df_carbs.set_index('date', inplace=True)

        # Dataframe workouts
        df_workouts = pd.DataFrame
        if 'activityName' in df.columns:
            df_workouts = df.copy()[df['activityName'].notna()]
            if not df_workouts.empty:
                if 'activityDuration.value' in df_workouts.columns:
                    df_workouts.rename(columns={"time": "date", "activityName": "workout_label"}, inplace=True)
                    df_workouts.sort_values(by='date', inplace=True, ascending=True)
                    # We need to split each sample into five minute intervals, and create new rows for each five minutes
                    expanded_rows = []
                    for _, row in df_workouts.iterrows():
                        intervals = split_workouts_into_intervals(row)
                        expanded_rows.extend(intervals)
                    df_workouts = pd.DataFrame(expanded_rows)
                    df_workouts.set_index('date', inplace=True)
                else:
                    print("No duration registered for physical activity!")

        return df_glucose, df_bolus, df_basal, df_carbs, df_workouts


def split_basal_into_intervals(row):
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
        current = next_interval

    return rows


def split_workouts_into_intervals(row):
    start = row.date
    if not pd.isna(row['activityDuration.value']):
        end = start + timedelta(seconds=row['activityDuration.value'])
    else:
        end = start + timedelta(seconds=10)
    bin_value = row['workout_label']
    if 'energy.value' in row:
        total_calories_burned = row['energy.value']
    else:
        total_calories_burned = np.nan
    interval = timedelta(minutes=5)
    total_duration = (end - start).total_seconds()

    # Track rows for the new DataFrame
    rows = []
    current = start

    while current < end:
        next_interval = min(current + interval, end)

        proportion = (next_interval - current).total_seconds() / total_duration
        calories_burned = total_calories_burned * proportion

        # Add the row to the list
        rows.append({
            'date': current,
            'workout_label': bin_value,
            'calories_burned': calories_burned
        })
        current = next_interval
    return rows


def get_dfs_and_ids(file_path, all_dfs, all_ids, is_test_bools, is_test, id_prefix):
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.csv'):
                subject_file_path = os.path.join(root, file)
                df = pd.read_csv(subject_file_path, low_memory=False)
                all_dfs.append(df)

                subject_id = id_prefix + file.split("_")[1].split(".")[0]
                all_ids.append(subject_id)

                is_test_bools.append(is_test)
    return all_dfs, all_ids, is_test_bools
