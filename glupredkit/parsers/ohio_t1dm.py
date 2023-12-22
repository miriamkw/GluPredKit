"""
The Ohio T1DM parser is processing the raw .xml data from the Ohio T1DM datasets and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import xml.etree.ElementTree as ET
import pandas as pd


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, subject_id: str, *args):
        """
        file_path -- the file path to where the test- and train-folder is located.
        id -- the id of the subject.
        """
        training_tree = ET.parse(file_path + f'train/{subject_id}-ws-training.xml')
        testing_tree = ET.parse(file_path + f'test/{subject_id}-ws-testing.xml')

        df_training = self.resample_data(training_tree)
        df_training['is_test'] = False

        df_testing = self.resample_data(testing_tree)
        df_testing['is_test'] = True

        merged_df = pd.concat([df_testing, df_training], ignore_index=False)

        # Sort the merged DataFrame
        merged_df = merged_df.sort_index()

        return merged_df

    def resample_data(self, tree):
        root = tree.getroot()

        dataframes = {}
        for child in root:
            tag_name = child.tag
            events = []
            for event in child.findall('event'):
                events.append(event.attrib)
            dataframes[tag_name] = pd.DataFrame(events)

        # Resampling all datatypes into the same time-grid
        df = dataframes['glucose_level'].copy()
        df['ts'] = pd.to_datetime(df['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.rename(columns={'value': 'CGM', 'ts': 'date'}, inplace=True)
        df.set_index('date', inplace=True)
        df = df.resample('5T', label='right').last()

        # Carbohydrates
        df_carbs = dataframes['meal'].copy()
        df_carbs['ts'] = pd.to_datetime(df_carbs['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df_carbs['carbs'] = pd.to_numeric(df_carbs['carbs'], errors='coerce')
        df_carbs.rename(columns={'ts': 'date'}, inplace=True)
        df_carbs = df_carbs[['date', 'carbs']]
        df_carbs.set_index('date', inplace=True)
        df_carbs = df_carbs.resample('5T', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_carbs, on="date", how='outer')
        df['carbs'] = df['carbs'].fillna(value=0.0)

        # Bolus doses
        df_bolus = dataframes['bolus'].copy()
        df_bolus['ts_begin'] = pd.to_datetime(df_bolus['ts_begin'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df_bolus['dose'] = pd.to_numeric(df_bolus['dose'], errors='coerce')
        df_bolus.rename(columns={'ts_begin': 'date', 'dose': 'bolus'}, inplace=True)
        df_bolus = df_bolus[['date', 'bolus']]
        df_bolus.set_index('date', inplace=True)
        df_bolus = df_bolus.resample('5T', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_bolus, on="date", how='outer')
        df['bolus'] = df['bolus'].fillna(value=0.0)

        # Basal rates
        df_basal = dataframes['basal'].copy()
        df_basal['ts'] = pd.to_datetime(df_basal['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df_basal['value'] = pd.to_numeric(df_basal['value'])
        df_basal.rename(columns={'ts': 'date', 'value': 'basal'}, inplace=True)
        df_basal = df_basal[['date', 'basal']]
        df_basal.set_index('date', inplace=True)
        df_basal = df_basal.resample('5T', label='right').asfreq().ffill()

        # Temp basal rates
        df_temp_basal = dataframes['temp_basal'].copy()
        if not df_temp_basal.empty:
            df_temp_basal['ts_begin'] = pd.to_datetime(df_temp_basal['ts_begin'], format='%d-%m-%Y %H:%M:%S',
                                                       errors='coerce')
            df_temp_basal['ts_end'] = pd.to_datetime(df_temp_basal['ts_end'], format='%d-%m-%Y %H:%M:%S',
                                                     errors='coerce')
            # Override the basal rates with the temp basal rate data
            for index, row in df_temp_basal.iterrows():
                start_date = row['ts_begin']  # Assuming the column name in df_temp_basal is 'start_date'
                end_date = row['ts_end']  # Assuming the column name in df_temp_basal is 'end_date'
                value = row['value']
                df_basal.loc[start_date:end_date] = float(value)

        # Convert basal rates from U/hr to U
        df_basal['basal'] = pd.to_numeric(df_basal['basal'], errors='coerce')
        df = pd.merge(df, df_basal, on="date", how='outer')
        df['basal'] = df['basal'].fillna(value=0.0)



        # Heart rate
        df_heartrate = dataframes['basis_heart_rate'].copy()
        if not df_heartrate.empty:
            df_heartrate['ts'] = pd.to_datetime(df_heartrate['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
            df_heartrate['value'] = pd.to_numeric(df_heartrate['value'], errors='coerce')
            df_heartrate.rename(columns={'ts': 'date', 'value': 'heartrate'}, inplace=True)
            df_heartrate.set_index('date', inplace=True)
            df_heartrate = df_heartrate.resample('5T', label='right').last()
            df = pd.merge(df, df_heartrate, on="date", how='outer')

        # Exercise
        df_exercise = dataframes['exercise'].copy()
        if not df_exercise.empty:
            df_exercise['ts'] = pd.to_datetime(df_exercise['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
            df_exercise['intensity'] = pd.to_numeric(df_exercise['intensity'], errors='coerce')
            df_exercise['duration'] = pd.to_numeric(df_exercise['duration'], errors='coerce')
            df_exercise['end_date'] = df_exercise['ts'] + pd.to_timedelta(df_exercise['duration'], unit='m')
            df_exercise.rename(columns={'ts': 'start_date', 'intensity': 'exercise'}, inplace=True)
            df['exercise'] = 0
            for idx, row in df_exercise.iterrows():
                # Find the range in df that falls between start_date and end_date
                mask = (df.index >= row['start_date']) & (df.index <= row['end_date'])
                df.loc[mask, 'exercise'] = row['exercise']

        df = df.sort_index()

        return df

