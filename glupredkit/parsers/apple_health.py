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
        unit = df.iloc[0]['unit']
        if str(unit).startswith('mmol'):
            df['value'] = df['value'].apply(lambda x: x * 18.018)

    df.rename(columns={'value': name}, inplace=True)
    df.rename(columns={'startDate': 'date'}, inplace=True)
    df.drop(['type', 'sourceName', 'sourceVersion', 'unit', 'creationDate', 'device', 'endDate'], axis=1, inplace=True)
    df.set_index('date', inplace=True)
    return df


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, start_date, end_date, file_path: str):
        tree = ET.parse(file_path)
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]
        data = pd.DataFrame(record_list)

        # Dates from string to datetime
        for col in ['creationDate', 'startDate', 'endDate']:
            data[col] = pd.to_datetime(data[col], errors='coerce')

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
        df_steps = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierStepCount', 'steps')
        df_restingheartrate = get_data_frame_for_type(data, 'HKQuantityTypeIdentifierRestingHeartRate', 'restingheartrate')

        # TODO: Add workouts and sleep data

        # Resampling all datatypes into the same time-grid
        df = df_glucose.copy()
        df = df.resample('5T', label='right').mean()

        df_carbs = df_carbs.resample('5T', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_carbs, on="date", how='outer')
        df['carbs'] = df['carbs'].fillna(value=0.0)

        df_insulin = df_insulin.resample('5T', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_insulin, on="date", how='outer')
        df['insulin'] = df['insulin'].fillna(value=0.0)

        df_heartrate = df_heartrate.resample('5T', label='right').mean()
        df = pd.merge(df, df_heartrate, on="date", how='outer')

        df_heartratevariability = df_heartratevariability.resample('5T', label='right').mean()
        df = pd.merge(df, df_heartratevariability, on="date", how='outer')

        df_caloriesburned = df_caloriesburned.resample('5T', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_caloriesburned, on="date", how='outer')
        df['caloriesburned'] = df['caloriesburned'].fillna(value=0.0)

        df_respiratoryrate = df_respiratoryrate.resample('5T', label='right').mean()
        df = pd.merge(df, df_respiratoryrate, on="date", how='outer')

        df_steps = df_steps.resample('5T', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_steps, on="date", how='outer')
        df['steps'] = df['steps'].fillna(value=0.0)

        df_restingheartrate = df_restingheartrate.resample('5T', label='right').mean()
        df = pd.merge(df, df_restingheartrate, on="date", how='outer')

        # Add hour of day
        df['hour'] = df.index.hour

        return df
