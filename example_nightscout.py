import asyncio
import datetime
from aiohttp import ClientError, ClientConnectorError, ClientResponseError
import py_nightscout as nightscout
import pytz
import pandas as pd

NIGHTSCOUT_URL = 'https://diabetes.neethan.net/'
API_SECRET = 'wD6KB2HvJ5ZL3FZphKjbPLdHb5C1zEix'

"""
In this example we use nightscout to get most recent data and print prediction so we can compare to Loop in real-time.
"""

async def main():
    """Example of library usage."""
    try:
        if API_SECRET:
            # To use authentication, use yout api secret:
            api = nightscout.Api(NIGHTSCOUT_URL, api_secret=API_SECRET)
        else:
            # You can use the api without authentication:
            api = nightscout.Api(NIGHTSCOUT_URL)
        status = await api.get_server_status()
    except ClientResponseError as error:
        raise RuntimeError("Received ClientResponseError") from error
    except (ClientError, ClientConnectorError, TimeoutError, OSError) as error:
        raise RuntimeError("Received client error or timeout") from error

    """
    Steps:
    1) Load data into dataframes
    2) Create method in loop model to get _one_ prediction
    3) Print prediction and compare to loop. 
        - Total
        - Insulin
        - other factors
    """



    #### Glucose Values (SGVs) ####
    # Get last 10 entries:
    entries = await api.get_sgvs()

    # Dataframe Glucose
    # time | units | value |
    times = [entry.date for entry in entries]
    units = ['mmol/L' for _ in entries]
    values = [entry.sgv_mmol for entry in entries]
    df_glucose = pd.DataFrame({'time': times, 'units': units, 'value': values})
    print(df_glucose)

    ### Treatments ####
    # To fetch recent treatments (boluses, temp basals):
    treatments = await api.get_treatments()

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
            absorption_times.append(treatment.absorptionTime)
    df_carbs = pd.DataFrame({'time': times, 'units': units, 'value': values, 'absorption_time\[s]': absorption_times})
    print(df_carbs)

    # Dataframe Bolus
    # time | dose[IU]
    times = []
    doses = []
    for treatment in treatments:
        if 'Bolus' in treatment.eventType:
            times.append(treatment.timestamp)
            doses.append(treatment.insulin)
    df_bolus = pd.DataFrame({'time': times, 'dose[IU]': doses})
    print(df_bolus)

    # Dataframe Basal
    # time | duration[ms] | rate[IU] | delivery_type |
    times = []
    durations = []
    rates = []
    types = []
    for treatment in treatments:
        if 'Temp Basal' == treatment.eventType:
            times.append(treatment.timestamp)
            durations.append(treatment.duration * 60000) # From minutes to ms
            rates.append(treatment.rate)
            types.append('temp')
        elif 'Basal' in treatment.eventType:
            times.append(treatment.timestamp)
            durations.append(treatment.duration * 60000) # From minutes to ms
            rates.append(treatment.rate)
            types.append('basal')
    df_basal = pd.DataFrame({'time': times, 'duration[ms]': durations, 'rate[IU]': rates, 'delivery_type': types})
    print(df_basal)




asyncio.run(main())