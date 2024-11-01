"""
The T1DEXI parser is processing the .xpt data from the Ohio T1DM datasets and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import pandas as pd
import os
import xport
import numpy as np
import datetime


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the T1DEXI dataset root folder.
        """
        df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, df_device = self.get_dataframes(file_path)
        df_resampled = self.resample_data(df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict,
                                          df_device)

        return df_resampled

    def resample_data(self, df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, df_device):
        # There are 88 subjects on MDI in the dataset --> 502 - 88 = 414. In the youth version: 261 - 37 = 224.
        # Total: 763 with, 638 without MDI
        # We use only the subjects not on MDI in this parser
        subject_ids_not_on_mdi = df_device[df_device['DXTRT'] != 'MULTIPLE DAILY INJECTIONS']['USUBJID'].unique()

        processed_dfs = []

        for subject_id in subject_ids_not_on_mdi:
            df_subject = df_glucose[df_glucose['id'] == subject_id].copy()
            df_subject = df_subject.resample('5min', label='left').last()
            df_subject['id'] = subject_id
            df_subject.sort_index(inplace=True)

            # TODO: Use a LLM to compute the amount of carbohydrates in the meal
            df_subject_meals = df_meals[df_meals['id'] == subject_id].copy()
            if not df_subject_meals.empty:
                df_subject_meal_grams = df_subject_meals[['meal_grams']].resample('5min', label='right').sum()
                df_subject_meal_name = df_subject_meals[['meal_name']].resample('5min', label='right').agg(
                    lambda x: ', '.join(x))

                df_subject = pd.merge(df_subject, df_subject_meal_grams, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_meal_name, on="date", how='outer')

                # Fill NaN where numbers were turned to 0 or strings were turned to ''
                df_subject['meal_grams'] = df_subject['meal_grams'].replace(0, np.nan)
                df_subject['meal_name'] = df_subject['meal_name'].replace('', np.nan)
            else:
                df_subject['meal_grams'] = np.nan
                df_subject['meal_name'] = np.nan

            df_subject_bolus = df_bolus[df_bolus['id'] == subject_id].copy()
            if not df_subject_bolus.empty:
                df_subject_bolus = df_subject_bolus[['bolus']].resample('5min', label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_bolus, on="date", how='outer')
            else:
                df_subject['bolus'] = np.nan

            # TODO: Double check that this is good!
            df_subject_basal = df_basal[df_basal['id'] == subject_id].copy()
            if not df_subject_basal.empty:
                df_subject_basal = df_subject_basal[['basal']].resample('5min', label='right').last()
                df_subject = pd.merge(df_subject, df_subject_basal, on="date", how='outer')
                # Forward fill the basal flow rates so that they are present for every 5-minute interval
                df_subject['basal'].ffill(inplace=True)
            else:
                df_subject['basal'] = np.nan

            df_subject_exercise = df_exercise[df_exercise['id'] == subject_id].copy()
            if not df_subject_exercise.empty:
                df_subject_exercise['workout'] = df_subject_exercise.apply(
                    lambda row: row['workout'] if row['workout'].lower() == row['workout_description'].lower()
                    else f"{row['workout']} {row['workout_description']}", axis=1)

                df_subject_exercise_workout = df_subject_exercise[['workout']].resample('5min', label='right').agg(
                    lambda x: ', '.join(sorted(set(x))))
                df_subject_exercise_workout_intensity = df_subject_exercise[['workout_intensity']].resample('5min',
                                                                                                            label='right').mean()
                df_subject_exercise_workout_duration = df_subject_exercise[['workout_duration']].resample('5min',
                                                                                                          label='right').sum()
                df_subject = pd.merge(df_subject, df_subject_exercise_workout, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_exercise_workout_intensity, on="date", how='outer')
                df_subject = pd.merge(df_subject, df_subject_exercise_workout_duration, on="date", how='outer')

                def forward_fill_by_duration(df, col):
                    # Iterate over each row and forward fill based on the duration
                    for i, row in df.iterrows():
                        if not pd.isna(row['workout_duration']):
                            # Calculate how many rows to forward fill based on the duration
                            forward_fill_count = int(row['workout_duration'] / 5) - 1

                            # Forward fill workout value to the next 'forward_fill_count' rows
                            for j in range(1, forward_fill_count + 1):
                                if i + pd.Timedelta(minutes=j * 5) in df.index:
                                    df.at[i + pd.Timedelta(minutes=j * 5), col] = row[col]
                    return df

                df_subject = forward_fill_by_duration(df_subject, 'workout')
                df_subject = forward_fill_by_duration(df_subject, 'workout_intensity')

                # Fill NaN where strings were turned to ''
                df_subject['workout'] = df_subject['workout'].replace('', np.nan)

                df_subject.drop(columns=['workout_duration'])
            else:
                df_subject['workout'] = np.nan
                df_subject['workout_intensity'] = np.nan

            if subject_id in heartrate_dict and not heartrate_dict[subject_id].empty:
                df_subject_heartrate = heartrate_dict[subject_id]
                df_subject_heartrate = df_subject_heartrate[['heartrate']].resample('5min', label='right').mean()
                df_subject = pd.merge(df_subject, df_subject_heartrate, on="date", how='outer')
            else:
                print(f"No heartrate data for subject {subject_id}")
                df_subject['heartrate'] = np.nan

            # Some rows might have gotten added nan values for the subject id after resampling
            df_subject['id'] = subject_id

            processed_dfs.append(df_subject)

        df_final = pd.concat(processed_dfs)
        return df_final


    def get_dataframes(self, file_path):
        """
        Extract the data that we are interested in from the dataset, and process them into our format and into separate
        dataframes.
        """
        glucose_file_path = os.path.join(file_path, 'LB.xpt')
        meals_file_path = os.path.join(file_path, 'ML.xpt')
        doses_file_path = os.path.join(file_path, 'FACM.xpt')
        device_file_path = os.path.join(file_path, 'DX.xpt')  # For filtering out people on MDI
        exercise_file_path = os.path.join(file_path, 'PR.xpt')
        heartrate_file_path = os.path.join(file_path, 'VS.xpt')

        row_count = 0
        chunksize = 1000000

        heartrate_dict = {}

        # Processing by chunks because the heartrate file is so big and can create memory problems
        for chunk in pd.read_sas(heartrate_file_path, chunksize=chunksize):
            row_count += chunksize
            df_heartrate = chunk.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

            # For some reason, the heart rate mean gives more data in T1DEXI version, while not in T1DEXIP
            if 'T1DEXIP' in file_path:
                df_heartrate = df_heartrate[df_heartrate['VSTEST'] == 'Heart Rate']
            else:
                df_heartrate = df_heartrate[df_heartrate['VSTEST'] == 'Heart Rate, Mean']

            for subject_id in df_heartrate['USUBJID'].unique():
                filtered_rows = df_heartrate.copy()[df_heartrate['USUBJID'] == subject_id]

                filtered_rows['VSDTC'] = pd.to_datetime(filtered_rows['VSDTC'], unit='s')
                filtered_rows['VSSTRESN'] = pd.to_numeric(filtered_rows['VSSTRESN'])
                filtered_rows.rename(columns={'VSDTC': 'date', 'VSSTRESN': 'heartrate'}, inplace=True)
                filtered_rows = filtered_rows[['heartrate', 'date']]
                filtered_rows.set_index('date', inplace=True)
                filtered_rows = filtered_rows.resample('5min', label='right').mean()

                # If the USUBJID is already in the dictionary, append the new data
                if subject_id in heartrate_dict:
                    heartrate_dict[subject_id] = pd.concat([heartrate_dict[subject_id], filtered_rows])
                else:
                    # Otherwise, create a new DataFrame for this USUBJID
                    heartrate_dict[subject_id] = filtered_rows

        # Open and read the relevant .xpt files
        with open(glucose_file_path, 'rb') as file:
            df_glucose = xport.to_dataframe(file)

        with open(meals_file_path, 'rb') as file:
            df_meals = xport.to_dataframe(file)

        with open(doses_file_path, 'rb') as file:
            df_insulin = xport.to_dataframe(file)

        with open(device_file_path, 'rb') as file:
            df_device = xport.to_dataframe(file)

        with open(exercise_file_path, 'rb') as file:
            df_exercise = xport.to_dataframe(file)

        df_glucose['LBDTC'] = pd.to_datetime(df_glucose['LBDTC'], unit='s')
        df_glucose['LBSTRESN'] = pd.to_numeric(df_glucose['LBSTRESN'], errors='coerce')
        df_glucose.rename(columns={'LBSTRESN': 'CGM', 'LBDTC': 'date', 'USUBJID': 'id'}, inplace=True)
        df_glucose = df_glucose[df_glucose['LBSTRESU'] == 'mg/dL']
        df_glucose = df_glucose[['id', 'CGM', 'date']]
        df_glucose.set_index('date', inplace=True)

        df_meals['MLDTC'] = pd.to_datetime(df_meals['MLDTC'], unit='s')
        df_meals['MLDOSE'] = pd.to_numeric(df_meals['MLDOSE'], errors='coerce')
        df_meals.rename(
            columns={'MLDOSE': 'meal_grams', 'MLTRT': 'meal_name', 'MLCAT': 'meal_category', 'MLDTC': 'date',
                     'USUBJID': 'id'}, inplace=True)
        #df_meals = df_meals[['meal_grams', 'meal_name', 'meal_category', 'id', 'date']]
        df_meals = df_meals[['meal_grams', 'meal_name', 'id', 'date']]
        df_meals.set_index('date', inplace=True)

        df_insulin['FADTC'] = pd.to_datetime(df_insulin['FADTC'], unit='s')
        df_insulin['FASTRESN'] = pd.to_numeric(df_insulin['FASTRESN'], errors='coerce')
        df_insulin.rename(columns={'FASTRESN': 'dose', 'FAORRESU': 'unit', 'FADTC': 'date', 'USUBJID': 'id'},
                          inplace=True)
        # df_insulin = df_insulin[['dose', 'unit', 'id', 'date']]
        df_insulin.set_index('date', inplace=True)

        # Bolus doses are always in unit U
        df_bolus = df_insulin[df_insulin['FACAT'] == 'BOLUS'][['dose', 'id']]
        df_bolus.rename(columns={'dose': 'bolus'}, inplace=True)

        # Basal rates are either in flow rate U/hr or in U
        df_basal = df_insulin[df_insulin['FACAT'] == 'BASAL']
        # The basal is structured so that there is one sample for total basal units, and another with the same date stamp for the flow rate
        # So we just extract the flow rate. The two of them essentially contain the same information
        df_basal = df_basal[df_basal['unit'] == 'U/hr']
        df_basal = df_basal[['dose', 'unit', 'id', 'FATESTCD', 'FATEST', 'FAOBJ']]
        df_basal.rename(columns={'dose': 'basal'}, inplace=True)

        df_exercise['PRSTDTC'] = pd.to_datetime(df_exercise['PRSTDTC'], unit='s')

        if 'T1DEXIP' in file_path:
            exercise_map = {
                '': 0,
                'Low: Pretty easy': 1.7,
                'Mild: Working a bit': 3.3,
                'Feeling the burn - Mild': 5.0,
                'Moderate: Working to keep up': 6.7,
                'Heavy: Hard to keep going but did it': 8.3,
                'Exhaustive: Too tough/Had to stop': 10
            }
            df_exercise['EXCINTSY'] = df_exercise['EXCINTSY'].map(exercise_map)
        else:
            # Original values are 0, 1 and 2, but we add 1 and multiply with 3.3 so the scale is from 0-10
            df_exercise['EXCINTSY'] = pd.to_numeric(df_exercise['EXCINTSY'])
            df_exercise['EXCINTSY'] = df_exercise['EXCINTSY'] + 1
            df_exercise['EXCINTSY'] *= 3.3
        df_exercise.rename(
            columns={'PRSTDTC': 'date', 'USUBJID': 'id', 'PRCAT': 'workout', 'PRTRTC': 'workout_description',
                     'PLNEXDUR': 'workout_duration', 'EXCINTSY': 'workout_intensity'}, inplace=True)
        df_exercise = df_exercise[
            ['workout', 'workout_description', 'workout_duration', 'workout_intensity', 'id', 'date']]
        df_exercise.set_index('date', inplace=True)

        return df_glucose, df_meals, df_bolus, df_basal, df_exercise, heartrate_dict, df_device



