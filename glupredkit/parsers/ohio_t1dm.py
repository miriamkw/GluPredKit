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

    def __call__(self, start_date, end_date, file_path: str):
        tree = ET.parse(file_path)
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
        df = df.resample('5T', label='right').mean()#.ffill(limit=1)

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
        df_basal['basal'] = df_basal['basal'] / 12
        df_basal.rename(columns={'basal': 'insulin'}, inplace=True)
        df_bolus.rename(columns={'bolus': 'insulin'}, inplace=True)

        df_insulin = df_basal.add(df_bolus, fill_value=0)
        df = pd.merge(df, df_insulin, on="date", how='outer')
        df['insulin'] = df['insulin'].fillna(value=0.0)

        # Heart rate
        df_heartrate = dataframes['basis_heart_rate'].copy()
        df_heartrate['ts'] = pd.to_datetime(df_heartrate['ts'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df_heartrate['value'] = pd.to_numeric(df_heartrate['value'], errors='coerce')
        df_heartrate.rename(columns={'ts': 'date', 'value': 'heartrate'}, inplace=True)
        df_heartrate.set_index('date', inplace=True)
        df_heartrate = df_heartrate.resample('5T', label='right').mean()
        df = pd.merge(df, df_heartrate, on="date", how='outer')

        return df
