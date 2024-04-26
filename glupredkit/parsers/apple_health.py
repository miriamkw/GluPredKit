"""
The Apple Health parser is processing the raw .xml data from Apple Health export and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import xml.etree.ElementTree as ET
import pandas as pd


def get_data_frame_for_type(data, type, name):
    df = data[data.type == type]
    df = df.copy()

    # Converting to mg/dL if necessary
    if name == "CGM":
        try:
            unit = df.iloc[0]['unit']
            if str(unit).startswith('mmol'):
                df['value'] = df['value'].apply(lambda x: x * 18.018)
        except IndexError as ie:
            print(f"Error: There are no blood glucose values in the dataset. Make sure to specify the correct start "
                  f"date in the parser. {ie}")
            raise
        except KeyError as ke:
            print(f"Error: The key '{ke}' does not exist in the DataFrame.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    df.rename(columns={'value': name}, inplace=True)
    df.rename(columns={'startDate': 'date'}, inplace=True)
    df.drop(['type', 'sourceName', 'sourceVersion', 'unit', 'creationDate', 'device', 'endDate'], axis=1, inplace=True)
    df.set_index('date', inplace=True)
    return df


def add_activity_states(df, start_date, end_date, workout_type):
    start_date = pd.to_datetime(start_date).tz_localize(df.index.tz)
    end_date = pd.to_datetime(end_date).tz_localize(df.index.tz)
    date_mask = df.index.to_series().between(start_date, end_date)
    df.loc[date_mask, 'activity_state'] = workout_type


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, start_date, end_date, file_path: str):
        tree = ET.parse(file_path)
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]
        data = pd.DataFrame(record_list)

        # Workout data
        workout_list = [x.attrib for x in root.iter('Workout')]
        workout_data = pd.DataFrame(workout_list)
        workout_data['workoutActivityType'] = (workout_data['workoutActivityType']
                                               .str.replace('HKWorkoutActivityType', ''))

        # Dates from string to datetime
        for col in ['creationDate', 'startDate', 'endDate']:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            workout_data[col] = pd.to_datetime(workout_data[col], errors='coerce')

        # value is numeric, NaN if fails
        data['value'] = pd.to_numeric(data['value'], errors='coerce')

        # Filter the data on the input start- and enddate
        data['startDate'] = data['startDate'].dt.tz_localize(None)
        data = data[(data['startDate'] >= start_date) & (data['startDate'] <= end_date)]

        # Creating separate dataframes for each data type
        df_glucose = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierBloodGlucose', 'CGM')
        df_insulin = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierInsulinDelivery', 'insulin')
        df_carbs = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierDietaryCarbohydrates', 'carbs')
        df_heartrate = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierHeartRate', 'heartrate')
        df_heartratevariability = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN',
                                                      'heartratevariability')
        df_caloriesburned = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierActiveEnergyBurned', 'caloriesburned')
        df_respiratoryrate = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierRespiratoryRate', 'respiratoryrate')
        df_vo2max = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierVO2Max', 'vo2max')
        df_steps = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierStepCount', 'steps')
        df_restingheartrate = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierRestingHeartRate', 'restingheartrate')

        # Resampling all datatypes into the same time-grid
        df = df_glucose.copy()
        df = df.resample('5min', label='right').mean()

        df_carbs = df_carbs.resample('5min', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_carbs, on="date", how='outer')
        df['carbs'] = df['carbs'].fillna(value=0.0)

        df_insulin = df_insulin.resample('5min', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_insulin, on="date", how='outer')
        df['insulin'] = df['insulin'].fillna(value=0.0)

        df_heartrate = df_heartrate.resample('5min', label='right').mean()
        df = pd.merge(df, df_heartrate, on="date", how='outer')

        df_heartratevariability = df_heartratevariability.resample('5min', label='right').mean()
        df = pd.merge(df, df_heartratevariability, on="date", how='outer')

        df_caloriesburned = df_caloriesburned.resample('5min', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_caloriesburned, on="date", how='outer')
        df['caloriesburned'] = df['caloriesburned'].fillna(value=0.0)

        df_respiratoryrate = df_respiratoryrate.resample('5min', label='right').mean()
        df = pd.merge(df, df_respiratoryrate, on="date", how='outer')

        df_vo2max = df_vo2max.resample('5min', label='right').mean()
        df = pd.merge(df, df_vo2max, on="date", how='outer')

        df_steps = df_steps.resample('5min', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_steps, on="date", how='outer')
        df['steps'] = df['steps'].fillna(value=0.0)

        df_restingheartrate = df_restingheartrate.resample('5min', label='right').mean()
        df = pd.merge(df, df_restingheartrate, on="date", how='outer')

        # Add workouts
        df['activity_state'] = "None"
        # Loop through each row in workout_data and call add_activity_states
        for index, row in workout_data.iterrows():
            add_activity_states(df, row['startDate'], row['endDate'], row['workoutActivityType'])

        # Add hour of day
        df['hour'] = df.index.hour

        return df
