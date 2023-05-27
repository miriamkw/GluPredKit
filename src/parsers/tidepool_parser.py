"""
The tidepool parser uses tidepool API to fetch some data using user credentials
and return the data in a format that can be used as input to the blood glucose prediction models.
"""
from tidepool_data_science_project.makedata.tidepool_api import TidepoolAPI
from datetime import datetime
import pandas as pd
from src.parsers.base_parser import BaseParser

class TidepoolParser(BaseParser):
    def __init__(self):
        super().__init__

    def __call__(self, start_date: datetime, end_date: datetime, username: str, password: str):
        """
        Tidepool API ignores time of day in the dates and will always fetch all data from a specific date
        """
        try:
            tp_api = TidepoolAPI(username, password)
            tp_api.login()

            # All the data in json format
            user_data = tp_api.get_user_event_data(start_date, end_date)

            tp_api.logout()

            # Sort data into one dataframe each data type, and retreive only the essential information
            (glucose_data, bolus_data, basal_data, carb_data) = self.parse_json(user_data)

            df_glucose = pd.json_normalize(glucose_data)[['time', 'units', 'value', 'origin.payload.sourceRevision.source.name']]
            df_bolus = pd.json_normalize(bolus_data)[['time', 'normal', 'origin.payload.device.name']]
            df_basal = pd.json_normalize(basal_data)[['time', 'duration', 'rate', 'origin.payload.device.name',
                                                      'payload.com.loopkit.InsulinKit.MetadataKeyScheduledBasalRate',
                                                      'deliveryType'
                                                      ]]
            df_carbs = pd.json_normalize(carb_data)[['time', 'nutrition.carbohydrate.units','nutrition.carbohydrate.net', 'payload.com.loopkit.AbsorptionTime']]

            # Rename columns
            df_glucose.rename(columns={"origin.payload.sourceRevision.source.name": "device_name"}, inplace=True)
            df_bolus.rename(columns={"normal": "dose[IU]", "origin.payload.device.name": "device_name"}, inplace=True)
            df_basal.rename(columns={"duration": "duration[ms]", "rate": "rate[U/hr]", "origin.payload.device.name": "device_name",
                                     "payload.com.loopkit.InsulinKit.MetadataKeyScheduledBasalRate": "scheduled_basal", "deliveryType": "delivery_type"}, inplace=True)
            df_carbs.rename(columns={"nutrition.carbohydrate.units": "units", "nutrition.carbohydrate.net": "value", "payload.com.loopkit.AbsorptionTime": "absorption_time[s]"}, inplace=True)

            # If blood glucose values in mmol/L, convert to mg/dL
            if not df_glucose.empty:
                if df_glucose.loc[0, 'units'] == 'mmol/L':
                    df_glucose['value'] = df_glucose['value'] * 18.0182
                    df_glucose['units'] = 'mg/dL'

            # Convert time to datetime object
            for df in [df_glucose, df_bolus, df_basal, df_carbs]:
                df.time = pd.to_datetime(df.time)

            return df_glucose, df_bolus, df_basal, df_carbs

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return []

    def parse_json(self, user_data):
        """
        Sort the user data from tidepool API into lists of the different data types

        Arguments:
        user_data -- the dictionary from an API call to the Tidepool data

        Output:
        Lists of the different data types in the json(glucose_data, bolus_data, basal_data, carb_data)
        """
        glucose_data = []
        bolus_data = []
        basal_data = []
        carb_data = []

        # Sort data types into lists
        for data in user_data:
            if data['type'] == 'cbg':
                glucose_data.append(data)
            elif data['type'] == 'bolus':
                bolus_data.append(data)
            elif data['type'] == 'basal':
                basal_data.append(data)
            elif data['type'] == 'food':
                carb_data.append(data)
            else:
                print('Unknown type is not yet supported: ', data['type'])

        return (glucose_data, bolus_data, basal_data, carb_data)
