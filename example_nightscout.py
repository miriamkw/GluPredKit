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
    # Get entries from the last six hours
    # The predictions will not be correct with too few values, but the exact number is currently unknown
    entries = api.get_sgvs({'count': 12*6})

    # Dataframe Glucose
    # time | units | value |
    times = [entry.date for entry in entries]
    units = ['mg/dL' for _ in entries]
    values = [entry.sgv for entry in entries]
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
            durations.append(treatment.duration * 60000) # From minutes to ms
            rates.append(treatment.rate)
            types.append('temp')
        elif 'Basal' in treatment.eventType:
            times.append(treatment.timestamp)
            durations.append(treatment.duration * 60000) # From minutes to ms
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
    recommendations = model.get_prediction_output(df_glucose, df_bolus, df_basal, df_carbs)
    inputs = recommendations.get("input_data")

    glucose_dates = recommendations.get("predicted_glucose_dates")[:73]
    glucose_values = [val / 18.0182 for val in recommendations.get("predicted_glucose_values")[:73]]

    start_date = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    start_glucose = inputs.get("glucose_values")[-1]

    (carb_dates,
     carb_values
     ) = predict_glucose(
        start_date, start_glucose,
        carb_effect_dates=recommendations.get("carb_effect_dates"),
        carb_effect_values=recommendations.get("carb_effect_values")
    )
    carb_values = [val / 18.0182 for val in carb_values]

    (insulin_dates,
     insulin_values
     ) = predict_glucose(
        start_date, start_glucose,
        insulin_effect_dates=recommendations.get("insulin_effect_dates"),
        insulin_effect_values=recommendations.get("insulin_effect_values")
    )
    insulin_values = [val / 18.0182 for val in insulin_values]

    (momentum_dates,
     momentum_values
     ) = predict_glucose(
        start_date, start_glucose,
        momentum_dates=recommendations.get("momentum_effect_dates"),
        momentum_values=recommendations.get("momentum_effect_values")
    )
    momentum_values = [val / 18.0182 for val in momentum_values]

    if recommendations.get("retrospective_effect_dates"):
        (retrospective_dates,
         retrospective_values
         ) = predict_glucose(
            start_date, start_glucose,
            correction_effect_dates=recommendations.get(
                "retrospective_effect_dates"
            ),
            correction_effect_values=recommendations.get(
                "retrospective_effect_values"
            )
        )
    else:
        (retrospective_dates,
         retrospective_values
         ) = ([], [])
    retrospective_values = [val / 18.0182 for val in retrospective_values]

    print("Start glucose: ", start_glucose / 18.0182)
    print("Final prediction: ", glucose_values[-1])
    print("")
    print("Final insulin: ", insulin_values[-1])
    print("Final carbs: ", carb_values[-1])
    if not len(momentum_values) == 0:
        print("Final momentum: ", momentum_values[-1])
    if not len(retrospective_values) == 0:
        print("Final retrospective: ", retrospective_values[-1])

    fig, ax = plt.subplots()
    ax.scatter(df_glucose.time, df_glucose.value, color='blue', label='True')
    ax.plot(glucose_dates, glucose_values, linestyle='--', color='blue', label='Predicted')
    ax.plot(carb_dates[:73], carb_values[:73], label='Carbohydrates', linestyle='--', color='orange')
    ax.plot(insulin_dates[:73], insulin_values[:73], label='Insulin', linestyle='--')
    ax.plot(momentum_dates, momentum_values, label='Momentum', linestyle='--')
    if len(retrospective_values) == 0:
        print("No retrospective values")
    else:
        ax.plot(retrospective_dates, retrospective_values, label='Retrospective', linestyle='--')

    plt.axhspan(5.6, 6.3, facecolor='b', alpha=0.2)

    ax.set(xlabel='Time (minutes)', ylabel='Blood Glucose (mg/dL)', title='Measured vs one trajectory of predicted values')
    ax.grid()
    plt.legend(loc='best')

    plt.show()

asyncio.run(main())