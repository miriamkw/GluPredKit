import asyncio
from datetime import timedelta, datetime
import pytz
from src.models.loop_model import LoopModel
from pyloopkit.loop_math import predict_glucose
import json
import matplotlib.pyplot as plt
from src.parsers.nightscout_parser import NightscoutParser

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
NIGHTSCOUT_URL = credentials['nightscout_api']['url']
API_SECRET = credentials['nightscout_api']['api_secret']

"""
In this example we use nightscout to get most recent data and print prediction so we can compare to Loop in real-time.
"""

async def main():

    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    parser = NightscoutParser()
    df_glucose, df_bolus, df_basal, df_carbs = parser(start_date=start_date, end_date=end_date, nightscout_url=NIGHTSCOUT_URL, api_secret=API_SECRET)

    print(df_glucose)

    print(df_carbs)

    print(df_basal)



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