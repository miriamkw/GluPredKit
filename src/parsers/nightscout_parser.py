"""
The nightscout parser uses nightscout API to fetch some data using user credentials
and return the data in a format that can be used as input to the blood glucose prediction models.
"""
from aiohttp import ClientError, ClientConnectorError, ClientResponseError
import nightscout
from src.parsers.base_parser import BaseParser
import pandas as pd
class NightscoutParser(BaseParser):
    def __init__(self):
        super().__init__

    def __call__(self, start_date, end_date, nightscout_url: str, api_secret: str, scheduled_basal=0.7):
        """
        Tidepool API ignores time of day in the dates and will always fetch all data from a specific date
        """
        try:
            api = nightscout.Api(nightscout_url, api_secret=api_secret)

            """
                Steps:
                1) Load data into dataframes
                2) Create method in loop model to get _one_ prediction
                3) Print prediction and compare to loop. 
                    - Total
                    - Insulin
                    - other factors
            """

            api_start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            api_end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

            # gte = greater than or equal to, lte = lower than or equal to
            query_params = {'count': 0}
            query_params['find[dateString][$gte]'] = api_start_date
            query_params['find[dateString][$lte]'] = api_end_date

            #### Glucose Values (SGVs) ####
            entries = api.get_sgvs(query_params)

            # Dataframe Glucose
            # time | units | value |
            times = [entry.date for entry in entries]
            units = ['mg/dL' for _ in entries]
            values = [entry.sgv for entry in entries]
            df_glucose = pd.DataFrame({'time': times, 'units': units, 'value': values})

            ### Treatments ####
            # To fetch recent treatments (boluses, temp basals):
            # By default returns the samples from the last 24 hours
            # DISCLAIMER: Sceduled basal rates might not be written to ns
            # DISCLAIMER2: Basal rates are of unit U/hr, and the predictions seem to be correct this way
            query_params = {'count': 0}
            query_params['find[timestamp][$gte]'] = api_start_date
            query_params['find[timestamp][$lte]'] = api_end_date

            treatments = api.get_treatments(query_params)

            # Dataframe Carbs
            # time | units | value | absorption_time\[s] |
            times = []
            units = []
            values = []
            absorption_times = []
            for treatment in treatments:
                if treatment.eventType == 'Carb Correction':
                    times.append(treatment.timestamp)
                    units.append('grams')
                    values.append(treatment.carbs)
                    absorption_times.append(treatment.absorptionTime * 60)
            df_carbs = pd.DataFrame(
                {'time': times, 'units': units, 'value': values, 'absorption_time[s]': absorption_times})

            # Dataframe Bolus
            # time | dose[IU]
            times = []
            doses = []
            for treatment in treatments:
                if 'Bolus' in treatment.eventType:
                    times.append(treatment.timestamp)
                    doses.append(treatment.insulin)
            df_bolus = pd.DataFrame({'time': times, 'dose[IU]': doses})

            # Dataframe Basal
            # time | duration[ms] | rate[U/hr] | delivery_type |
            times = []
            durations = []
            rates = []
            types = []
            for treatment in treatments:
                if 'Temp Basal' == treatment.eventType:
                    times.append(treatment.timestamp)
                    durations.append(treatment.duration * 60000)  # From minutes to ms
                    rates.append(treatment.rate)
                    types.append('temp')
                elif 'Basal' in treatment.eventType:
                    times.append(treatment.timestamp)
                    durations.append(treatment.duration * 60000)  # From minutes to ms
                    rates.append(treatment.rate)
                    types.append('basal')
            df_basal = pd.DataFrame(
                {'time': times, 'duration[ms]': durations, 'rate[U/hr]': rates, 'delivery_type': types})

            # ADDING SHEDULED BASAL RATES BECAUSE NIGHTSCOUT DOES NOT REGISTER THIS AS A TREATMENT

            # Convert the "date" column to datetime
            df_basal['time'] = pd.to_datetime(df_basal['time'])

            # Convert the "duration" column from milliseconds to seconds
            df_basal['duration[ms]'] = df_basal['duration[ms]'] / 1000

            # Define the tolerance value for the time difference
            tolerance = 1  # second

            # Create an empty dataframe to store the new rows
            new_rows = pd.DataFrame(columns=df_basal.columns)

            # Loop through the rows of the dataframe and check for gaps
            for i, row in df_basal.iterrows():
                if i == len(df_basal) - 1:  # last row
                    continue

                next_row = df_basal.iloc[i + 1]
                time_diff = (row['time'] - next_row['time']).total_seconds()
                if time_diff > (next_row['duration[ms]'] + tolerance):
                    # There is a gap, add a new row with default value
                    new_date = next_row['time'] + pd.Timedelta(milliseconds=row['duration[ms]'] * 1000)
                    new_duration = time_diff - next_row['duration[ms]']
                    new_row = pd.DataFrame([[new_date, new_duration, scheduled_basal, 'temp']], columns=df_basal.columns)
                    new_rows = new_rows.append(new_row, ignore_index=True)

            # Sort the dataframe in ascending order based on "date"
            df_basal = df_basal.sort_values(by='time', ascending=False)

            # Convert duration back to ms
            df_basal['duration[ms]'] = df_basal['duration[ms]'] * 1000

            return df_glucose, df_bolus, df_basal, df_carbs

        except ClientResponseError as error:
            raise RuntimeError("Received ClientResponseError") from error
        except (ClientError, ClientConnectorError, TimeoutError, OSError) as error:
            raise RuntimeError("Received client error or timeout") from error
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return []

