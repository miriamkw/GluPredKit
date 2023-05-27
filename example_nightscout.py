import asyncio
from datetime import timedelta, datetime
from src.models.loop_model import LoopModel
from src.plots.loop_trajectories import LoopTrajectories
import json
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

    model = LoopModel()
    loop_model_output = model.get_prediction_output(df_glucose, df_bolus, df_basal, df_carbs)

    plot = LoopTrajectories()
    plot._draw_plot(loop_model_output, glucose_unit='mg/dL')


asyncio.run(main())