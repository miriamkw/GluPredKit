from src.plots.interactive_prediction import InteractivePrediction
import asyncio
import datetime
from aiohttp import ClientError, ClientConnectorError, ClientResponseError
import nightscout
import pytz
import pandas as pd
from src.models.loop_model import LoopModel
from pyloopkit.loop_math import predict_glucose
import json
import matplotlib.pyplot as plt

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
NIGHTSCOUT_URL = credentials['nightscout_api']['url']
API_SECRET = credentials['nightscout_api']['api_secret']

try:
    if API_SECRET:
        # To use authentication, use yout api secret:
        api = nightscout.Api(NIGHTSCOUT_URL, api_secret=API_SECRET)
    else:
        # You can use the api without authentication:
        api = nightscout.Api(NIGHTSCOUT_URL)
except ClientResponseError as error:
    raise RuntimeError("Received ClientResponseError") from error
except (ClientError, ClientConnectorError, TimeoutError, OSError) as error:
    raise RuntimeError("Received client error or timeout") from error

#### Glucose Values (SGVs) ####
# Get entries from the last six hours
# The predictions will not be correct with too few values, but the exact number is currently unknown
entries = api.get_sgvs({'count': 12 * 6})

# Dataframe Glucose
# time | units | value |
times = [entry.date for entry in entries]
units = ['mmol/L' for _ in entries]
values = [entry.sgv / 18.0182 for entry in entries]
df_glucose = pd.DataFrame({'time': times, 'units': units, 'value': values})
print(df_glucose)

### Treatments ####
# To fetch recent treatments (boluses, temp basals):
# By default returns the samples from the last 24 hours
# DISCLAIMER: Sceduled basal rates might not be written to ns
# DISCLAIMER2: Basal rates are of unit U/hr, and the predictions seem to be correct this way
treatments = api.get_treatments()

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
df_carbs = pd.DataFrame({'time': times, 'units': units, 'value': values, 'absorption_time[s]': absorption_times})
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
df_basal = pd.DataFrame({'time': times, 'duration[ms]': durations, 'rate[U/hr]': rates, 'delivery_type': types})

# ADDING SHEDULED BASAL RATES AS NIGHTSCOUT DOES NOT REGISTER THIS AS A TREATMENT

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
        new_row = pd.DataFrame([[new_date, new_duration, 0.7, 'temp']], columns=df_basal.columns)
        new_rows = new_rows.append(new_row, ignore_index=True)

# Sort the dataframe in ascending order based on "date"
df_basal = df_basal.sort_values(by='time', ascending=False)

# Convert duration back to ms
df_basal['duration[ms]'] = df_basal['duration[ms]'] * 1000

model = LoopModel()

plot = InteractivePrediction()
plot(model, df_glucose, df_bolus, df_basal, df_carbs)
