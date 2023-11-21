"""
The nightscout parser uses nightscout API to fetch some data using user credentials
and return the data in a format that can be used as input to the blood glucose prediction trained_models.
"""
from aiohttp import ClientError, ClientConnectorError, ClientResponseError
import nightscout
from .base_parser import BaseParser
import pandas as pd
import datetime


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, start_date, end_date, username: str, password: str):
        """
        In the nighscout parser, the username is the nightscout URL, and the password is the API key.
        """
        try:
            api = nightscout.Api(username, api_secret=password)

            api_start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            api_end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

            # gte = greater than or equal to, lte = lower than or equal to
            query_params = {'count': 0, 'find[dateString][$gte]': api_start_date,
                            'find[dateString][$lte]': api_end_date}

            # Dataframe Glucose (SGV) [mg/dL]
            entries = api.get_sgvs(query_params)
            dates = [entry.date for entry in entries]
            values = [entry.sgv for entry in entries]
            df_glucose = pd.DataFrame({'date': dates, 'CGM': values})
            df_glucose.sort_values(by='date', inplace=True, ascending=True)
            df_glucose.set_index('date', inplace=True)

            # Treatments call returns insulin (bolus and basal separated) and carbohydrate intakes
            query_params = {'count': 0, 'find[timestamp][$gte]': api_start_date, 'find[timestamp][$lte]': api_end_date}
            treatments = api.get_treatments(query_params)

            # Dataframe Carbs
            dates = []
            values = []
            for treatment in treatments:
                if treatment.eventType == 'Carb Correction':
                    dates.append(treatment.timestamp)
                    values.append(treatment.carbs)
            df_carbs = pd.DataFrame(
                {'date': dates, 'carbs': values})
            df_carbs.sort_values(by='date', inplace=True, ascending=True)
            df_carbs.set_index('date', inplace=True)

            # Dataframe Bolus
            dates = []
            values = []
            for treatment in treatments:
                if 'Bolus' in treatment.eventType:
                    dates.append(treatment.timestamp)
                    values.append(treatment.insulin)
            df_bolus = pd.DataFrame({'date': dates, 'insulin': values})
            df_bolus.sort_values(by='date', inplace=True, ascending=True)
            df_bolus.set_index('date', inplace=True)

            # Dataframe Basal rates [U/hr]
            dates = []
            values = []
            for treatment in treatments:
                if 'Temp Basal' == treatment.eventType:
                    dates.append(treatment.timestamp)
                    values.append(treatment.rate)
                elif 'Basal' in treatment.eventType:
                    dates.append(treatment.timestamp)
                    values.append(treatment.rate)
            df_basal = pd.DataFrame(
                {'date': dates, 'basal_rate': values})
            df_basal.sort_values(by='date', inplace=True, ascending=True)
            df_basal.set_index('date', inplace=True)

            # Resampling all datatypes into the same time-grid
            df = df_glucose.copy()
            df = df.resample('5T', label='right').mean()

            df_carbs = df_carbs.resample('5T', label='right').sum().fillna(value=0)
            df = pd.merge(df, df_carbs, on="date", how='outer')
            df['carbs'] = df['carbs'].fillna(value=0.0)

            df_bolus = df_bolus.resample('5T', label='right').sum()
            df = pd.merge(df, df_bolus, on="date", how='outer')

            # The reason why NS basal data is different from Tidepool and Apple health,
            # Is that the stored data is not based on actually delivered basal units (like in Apple health)
            # Nor is it the derived temperal basal rate from the actually delivered doses (like in Tidepool),
            # But NS is based on the "programmed temperal basal rate (see metadata apple health),
            # Which is unfortunately quite unaccurate compared to the delivered doses from the pump.
            df_basal = df_basal.resample('5T', label='right').last()
            df_basal['basal_rate'] = df_basal['basal_rate'] / 60 * 5  # From U/hr to U (5-minutes)
            df = pd.merge(df, df_basal, on="date", how='outer')
            df['basal_rate'] = df['basal_rate'].ffill(limit=12 * 24 * 2) # Forward fill the basal rate
            df[['insulin', 'basal_rate']] = df[['insulin', 'basal_rate']].fillna(value=0.0)
            df['insulin'] = df['insulin'] + df['basal_rate']
            df.drop(columns=(["basal_rate"]), inplace=True)

            # Get the current datetime in UTC, given the calendar on current computer
            current_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            df.index = df.index.tz_convert(current_timezone)

            # Add hour of day
            df['hour'] = df.index.hour

            return df

        except ClientResponseError as error:
            raise RuntimeError("Received ClientResponseError") from error
        except (ClientError, ClientConnectorError, TimeoutError, OSError) as error:
            raise RuntimeError("Received client error or timeout. Make sure that the username (nightscout URL) and "
                               "passoword (API key) is correct.") from error
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {str(e)}")
