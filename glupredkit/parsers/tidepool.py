"""
The tidepool parser uses tidepool API to fetch some data using user credentials
and return the data in a format that can be used as input to the blood glucose prediction trained_models.
"""
from tidepool_data_science_project.makedata.tidepool_api import TidepoolAPI
import datetime
import json
import pandas as pd
from dateutil import parser
from .base_parser import BaseParser


class Parser(BaseParser):
    def __init__(self):
        super().__init__

    def __call__(self, start_date, end_date, username: str, password: str):
        """
        Tidepool API ignores time of day in the dates and will always fetch all data from a specific date
        """
        try:
            tp_api = TidepoolAPI(username, password)
            tp_api.login()

            # All the data in json format
            user_data = tp_api.get_user_event_data(start_date, end_date)

            tp_api.logout()

            # Sort data into one dataframe each data type, and retrieve only the essential information
            (glucose_data, bolus_data, basal_data, carb_data, workout_data) = self.parse_json(user_data)

            # Dataframe blood glucose
            df_glucose = pd.json_normalize(glucose_data)[['time', 'value']]
            if not df_glucose.empty:  # If blood glucose values in mmol/L, convert to mg/dL
                if pd.json_normalize(glucose_data)[['units']].loc[0, 'units'] == 'mmol/L':
                    df_glucose['value'] = df_glucose['value'] * 18.0182
            df_glucose.time = df_glucose.time.apply(self.custom_date_parser)
            df_glucose.time = pd.to_datetime(df_glucose.time, errors='coerce')
            df_glucose.rename(columns={"time": "date", "value": "CGM"}, inplace=True)
            df_glucose.sort_values(by='date', inplace=True, ascending=True)
            df_glucose.set_index('date', inplace=True)

            # Dataframe bolus doses
            df_bolus = pd.json_normalize(bolus_data)[['time', 'normal']]
            df_bolus.time = pd.to_datetime(df_bolus.time, errors='coerce')
            df_bolus.rename(columns={"time": "date", "normal": "insulin"}, inplace=True)
            df_bolus.sort_values(by='date', inplace=True, ascending=True)
            df_bolus.set_index('date', inplace=True)

            # Dataframe basal rates
            df_basal = pd.json_normalize(basal_data)[
                ['time', 'rate']]
            df_basal.time = pd.to_datetime(df_basal.time, errors='coerce')
            df_basal.rename(columns={"time": "date", "rate": "basal"}, inplace=True)
            df_basal.sort_values(by='date', inplace=True, ascending=True)
            df_basal.set_index('date', inplace=True)

            # Dataframe carbohydrates
            df_carbs = pd.json_normalize(carb_data)[['time', 'nutrition.carbohydrate.net']]
            df_carbs.time = pd.to_datetime(df_carbs.time, errors='coerce')
            df_carbs.rename(columns={"time": "date", "nutrition.carbohydrate.net": "carbs"}, inplace=True)
            df_carbs.sort_values(by='date', inplace=True, ascending=True)
            df_carbs.set_index('date', inplace=True)

            if not len(workout_data) == 0:
                df_workouts = pd.json_normalize(workout_data)[['time', 'duration.value', 'name']]
                df_workouts.rename(columns={"duration.value": "duration[s]"}, inplace=True)
                df_workouts['name'] = df_workouts['name'].apply(lambda x: x.split()[0])
                df_workouts.time = pd.to_datetime(df_workouts.time, errors='coerce')
            else:
                df_workouts = pd.DataFrame()

            # Resampling all datatypes into the same time-grid
            df = df_glucose.copy()
            df = df.resample('5min', label='right').mean()

            df_carbs = df_carbs.resample('5min', label='right').sum().fillna(value=0)
            df = pd.merge(df, df_carbs, on="date", how='outer')
            df['carbs'] = df['carbs'].fillna(value=0.0)

            df_bolus = df_bolus.resample('5min', label='right').sum()
            df = pd.merge(df, df_bolus, on="date", how='outer')
            df['bolus'] = df['insulin']

            # TODO: The accuracy of the basal delivery could be improved by using duration and ensuring the exact
            # TODO: sum of the basal delivery is used...
            df_basal = df_basal.resample('5min', label='right').last()
            df_basal['basal'] = df_basal['basal'] / 60 * 5  # From U/hr to U (5-minutes)
            df = pd.merge(df, df_basal, on="date", how='outer')
            df['basal'] = df['basal'].ffill(limit=12 * 24 * 2)
            df[['insulin', 'basal']] = df[['insulin', 'basal']].fillna(value=0.0)
            df['insulin'] = df['insulin'] + df['basal']

            if not df_workouts.empty:
                # Add activity states
                df['activity_state'] = "None"
                df_workouts.apply(lambda x: self.add_activity_states(x['time'], x['duration[s]'], x['name'], df),
                                  axis=1)

            # Get the current datetime in UTC, given the calendar on current computer
            current_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            df.index = df.index.tz_convert(current_timezone)

            df['id'] = 1

            return df

        except Exception as e:
            raise RuntimeError(f"Error fetching data: {str(e)}")

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
        workout_data = []

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
            elif data['type'] == 'physicalActivity':
                workout_data.append(data)
            else:
                print('Unknown type is not supported: ', data['type'])

        return glucose_data, bolus_data, basal_data, carb_data, workout_data

    def add_activity_states(self, start_date, duration, workout_name, df):
        end_date = start_date + pd.to_timedelta(duration, unit='s')
        #df['activity_state'][df.index.to_series().between(start_date, end_date)] = workout_name
        df.loc[df.index.to_series().between(start_date, end_date), 'activity_state'] = workout_name

    # Your custom parser function
    def custom_date_parser(self, date_str):
        try:
            # Try parsing with the first format
            return parser.parse(date_str)
        except ValueError:
            # If the first format fails, try the second format
            try:
                return parser.parse(date_str, fuzzy=True)
            except ValueError:
                # If second format also fails, return NaT
                return pd.NaT

    def get_json_from_parsed_df(self, start_date, end_date, username: str, password: str, start_test_date, basal,
                               isf, cr):
        df = self.__call__(start_date, end_date, username, password)

        # Dataframe blood glucose
        df_glucose = df['CGM'].copy()
        df_glucose = df_glucose[df_glucose.index <= start_test_date]
        df_glucose = df_glucose.reset_index()
        df_glucose['date'] = df_glucose['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_glucose['CGM'] = df_glucose['CGM'].astype(int)
        df_glucose.rename(columns={"CGM": "value"}, inplace=True)
        df_glucose.sort_values(by='date', inplace=True, ascending=True)

        # Dataframe carbohydrates
        df_carbs = df['carbs'].copy()
        df_carbs = df_carbs[df_carbs.index <= start_test_date]
        df_carbs = df_carbs.reset_index()
        df_carbs = df_carbs[df_carbs['carbs'] > 0]
        df_carbs['absorptionTime'] = 10800
        df_carbs.rename(columns={"carbs": "grams"}, inplace=True)
        df_carbs.sort_values(by='date', inplace=True, ascending=True)
        df_carbs['date'] = df_carbs['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Dataframe bolus doses
        df_bolus = df['bolus'].copy()
        df_bolus = df_bolus[df_bolus.index <= start_test_date]
        df_bolus = df_bolus.reset_index()
        df_bolus = df_bolus[df_bolus['bolus'] > 0]
        df_bolus.rename(columns={"date": "startDate", "bolus": "volume"}, inplace=True)
        df_bolus.sort_values(by='startDate', inplace=True, ascending=True)
        df_bolus['startDate'] = df_bolus['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_bolus['endDate'] = df_bolus['startDate']
        df_bolus['type'] = "bolus"

        # Dataframe basal rates
        df_basal = df['basal'].copy()
        df_basal = df_basal[df_basal.index <= start_test_date]
        df_basal = df_basal.reset_index()
        df_basal.rename(columns={"date": "startDate", "basal": "volume"}, inplace=True)
        df_basal.sort_values(by='startDate', inplace=True, ascending=True)
        df_basal['endDate'] = df_basal['startDate'] + pd.Timedelta(minutes=5)
        df_basal['startDate'] = df_basal['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_basal['endDate'] = df_basal['endDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_basal['type'] = "basal"

        df_insulin = pd.concat([df_bolus, df_basal], ignore_index=True)
        df_insulin.sort_values(by='startDate', inplace=True, ascending=True)

        insulin_history_list = df_insulin.to_dict(orient='records')
        glucose_history_list = df_glucose.to_dict(orient='records')
        carbs_history_list = df_carbs.to_dict(orient='records')

        basal = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": basal}]
        carb_ratio = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": cr}]
        ins_sens = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": isf}]

        # Create the final JSON structure
        data = {
            "carbEntries": carbs_history_list,
            "doses": insulin_history_list,
            "glucoseHistory": glucose_history_list,
            "basal": basal,
            "carbRatio": carb_ratio,
            "sensitivity": ins_sens
        }
        return json.dumps(data, indent=2)

    def get_json_from_raw_data(self, start_date, end_date, username: str, password: str, start_test_date, basal,
                               isf, cr, recommendation_settings=None):
        tp_api = TidepoolAPI(username, password)
        tp_api.login()

        # All the data in json format
        user_data = tp_api.get_user_event_data(start_date, end_date)

        tp_api.logout()

        # Sort data into one dataframe each data type, and retrieve only the essential information
        (glucose_data, bolus_data, basal_data, carb_data, workout_data) = self.parse_json(user_data)

        # Dataframe blood glucose
        df_glucose = pd.json_normalize(glucose_data)[['time', 'value']]
        if not df_glucose.empty:  # If blood glucose values in mmol/L, convert to mg/dL
            if pd.json_normalize(glucose_data)[['units']].loc[0, 'units'] == 'mmol/L':
                df_glucose['value'] = df_glucose['value'] * 18.0182
        df_glucose.time = df_glucose.time.apply(self.custom_date_parser)
        df_glucose = df_glucose[df_glucose.time <= start_test_date]
        df_glucose['time'] = df_glucose['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_glucose['value'] = df_glucose['value'].astype(int)
        df_glucose.rename(columns={"time": "date"}, inplace=True)
        df_glucose.sort_values(by='date', inplace=True, ascending=True)

        # Dataframe bolus doses
        df_bolus = pd.json_normalize(bolus_data)[['time', 'normal']]
        df_bolus.time = pd.to_datetime(df_bolus.time, errors='coerce')
        df_bolus = df_bolus[df_bolus.time <= start_test_date]
        df_bolus.rename(columns={"time": "startDate", "normal": "volume"}, inplace=True)
        df_bolus['endDate'] = df_bolus['startDate']
        df_bolus.sort_values(by='startDate', inplace=True, ascending=True)
        df_bolus['type'] = "bolus"
        df_bolus['startDate'] = df_bolus['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_bolus['endDate'] = df_bolus['endDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Dataframe basal rates
        df_basal = pd.json_normalize(basal_data)[['time', 'rate', 'duration']]
        df_basal['duration'] = df_basal['duration'] / 60000  # Minutes
        df_basal.time = pd.to_datetime(df_basal.time, errors='coerce')
        df_basal['duration'] = pd.to_numeric(df_basal['duration'])
        df_basal['endDate'] = df_basal.apply(lambda row: row['time'] + pd.Timedelta(minutes=row['duration']),
                                             axis=1)
        df_basal = df_basal[df_basal.time <= start_test_date]
        df_basal.rename(columns={"time": "startDate", "rate": "volume"}, inplace=True)
        df_basal.sort_values(by='startDate', inplace=True, ascending=True)
        df_basal['type'] = "basal"
        df_basal['startDate'] = df_basal['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_basal['endDate'] = df_basal['endDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_basal['volume'] = df_basal['volume'] * df_basal['duration'] / 60
        df_basal = df_basal.drop(columns=['duration'])

        # Dataframe carbohydrates
        df_carbs = pd.json_normalize(carb_data)[['time', 'nutrition.carbohydrate.net',
                                                 'payload.com.loopkit.AbsorptionTime']]
        df_carbs.time = pd.to_datetime(df_carbs.time, errors='coerce')
        df_carbs = df_carbs[df_carbs.time <= start_test_date]
        df_carbs.rename(columns={"time": "date", "nutrition.carbohydrate.net": "grams",
                                 "payload.com.loopkit.AbsorptionTime": "absorptionTime"}, inplace=True)
        df_carbs.sort_values(by='date', inplace=True, ascending=True)
        df_carbs['date'] = df_carbs['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        df_insulin = pd.concat([df_bolus, df_basal], ignore_index=True)
        df_insulin.sort_values(by='startDate', inplace=True, ascending=True)

        insulin_history_list = df_insulin.to_dict(orient='records')
        glucose_history_list = df_glucose.to_dict(orient='records')
        carbs_history_list = df_carbs.to_dict(orient='records')

        basal = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": basal}]
        carb_ratio = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": cr}]
        ins_sens = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": isf}]

        # Create the final JSON structure
        data = {
            "carbEntries": carbs_history_list,
            "doses": insulin_history_list,
            "glucoseHistory": glucose_history_list,
            "basal": basal,
            "carbRatio": carb_ratio,
            "sensitivity": ins_sens
        }

        if recommendation_settings:
            data = {**data, **recommendation_settings}

        # Convert the data dictionary to a JSON string
        return json.dumps(data, indent=2)

    def getRecommendationSettings(self):
        data = {
            "predictionStart": "2024-06-23T16:00:11Z",
            "maxBasalRate": 4.1,
            "maxBolus": 9,
            "target": [
                {
                    "endDate": "2025-10-18T03:10:00Z",
                    "lowerBound": 101,
                    "startDate": "2022-10-17T20:59:03Z",
                    "upperBound": 115
                }
            ],
            # Are these necessary?
            "recommendationInsulinType": "novolog",
            "recommendationType": "automaticBolus",
        }
        return data

