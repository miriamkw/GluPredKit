from aiohttp import ClientError, ClientConnectorError, ClientResponseError
from src.parsers.base_parser import BaseParser
import pandas as pd
import requests

class OuraRingParser(BaseParser):
    def __init__(self):
        super().__init__

    def __call__(self, start_date, end_date, access_token):
        try:
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            headers = {
                'Authorization': 'Bearer ' + access_token
            }

            url_readiness = 'https://api.ouraring.com/v2/usercollection/daily_readiness'
            r = requests.request('GET', url_readiness, headers=headers, params=params)
            data_readiness = r.json()
            df_readiness = pd.DataFrame(data_readiness["data"])[['day', 'score']]
            df_readiness.rename(columns={"score": "readiness_score"}, inplace=True)

            url_sleep = 'https://api.ouraring.com/v2/usercollection/daily_sleep'
            r = requests.request('GET', url_sleep, headers=headers, params=params)
            data_sleep = r.json()
            df_sleep = pd.DataFrame(data_sleep["data"])[['day', 'score']]
            df_sleep.rename(columns={"score": "sleep_score"}, inplace=True)

            df_oura = pd.merge(df_readiness, df_sleep, on='day', how='inner')

            return df_oura

        except ClientResponseError as error:
            raise RuntimeError("Received ClientResponseError") from error
        except (ClientError, ClientConnectorError, TimeoutError, OSError) as error:
            raise RuntimeError("Received client error or timeout") from error
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return []


