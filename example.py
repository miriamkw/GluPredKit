import asyncio
from datetime import timedelta, datetime
from src.models.linear_regressor import LinearRegressor
from src.plots.loop_trajectories import LoopTrajectories
from src.parsers.tidepool_parser import TidepoolParser
import json
import numpy as np
from src.parsers.nightscout_parser import NightscoutParser
from src.metrics.rmse import RMSE
from src.metrics.mae import MAE

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)

# Nightscout settings
NIGHTSCOUT_URL = credentials['nightscout_api']['url']
API_SECRET = credentials['nightscout_api']['api_secret']

# Tidepool settings
username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']


"""
In this example we use nightscout to get most recent data and print prediction so we can compare to Loop in real-time.
"""

async def main():

    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    parser = NightscoutParser()
    df = parser(start_date=start_date, end_date=end_date, nightscout_url=NIGHTSCOUT_URL, api_secret=API_SECRET)

    print(df)

    output_offset = 15

    # Train linear regression model
    model = LinearRegressor(output_offset=output_offset)
    model.fit(df)
    y_pred = model.predict(df)
    _, y_true = model.process_data(df)

    plot = LoopTrajectories()
    plot._draw_plot(y_pred, glucose_unit='mg/dL')


asyncio.run(main())


"""
Get tidepool data and predictions
"""
# Fetch training data from Tidepool API
start_date = datetime(2023, 3, 1)
end_date = datetime(2023, 3, 1)


tidepool_parser = TidepoolParser()

df_glucose, df_bolus, df_basal, df_carbs = tidepool_parser(start_date, end_date, username, password)

# Define the output offset to 30 minutes into the future
output_offset = 30

# Train linear regression model
model = LinearRegressor(output_offset=output_offset)
model.fit(df_glucose, df_bolus, df_basal, df_carbs)

# Get test data separate from training data and get predictions
start_date = datetime(2023, 3, 2)
end_date = datetime(2023, 3, 2)
df_glucose, df_bolus, df_basal, df_carbs = tidepool_parser(start_date, end_date, username, password)
y_pred = model.predict(df_glucose, df_bolus, df_basal, df_carbs)
_, y_true = model.process_data(df_glucose)

# Convert to mg/dL before fetching metrics
y_pred = np.multiply(y_pred, 18)
y_true = np.multiply(y_true, 18)

# Print different error metrics
rmse = RMSE()
mae = MAE()

print("RMSE: ", rmse(y_true, y_pred))
print("MAE: ", mae(y_true, y_pred))










