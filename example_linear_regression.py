import numpy as np

from src.parsers.tidepool_parser import fetch_tidepool_data
from src.models.linear_regressor import LinearRegressor
from datetime import datetime
from src.metrics.rmse import RMSE
from src.metrics.bayer import Bayer
from src.metrics.kovatchev import Kovatchev
import json

# Fetch training data from Tidepool API
start_date = datetime(2023, 3, 1)
end_date = datetime(2023, 3, 1)

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']

df_glucose, df_bolus, df_basal, df_carbs = fetch_tidepool_data(username, password, start_date, end_date)

# Define the output offset to 30 minutes into the future
output_offset = 30

# Train linear regression model
model = LinearRegressor(output_offset=output_offset)
model.fit(df_glucose, df_bolus, df_basal, df_carbs)

# Get test data separate from training data and get predictions
start_date = datetime(2023, 3, 2)
end_date = datetime(2023, 3, 2)
df_glucose, df_bolus, df_basal, df_carbs = fetch_tidepool_data(username, password, start_date, end_date)
y_pred, y_test = model.predict(df_glucose, df_bolus, df_basal, df_carbs)
y_test = y_test.to_numpy()

# Convert to mg/dL before fetching metrics
y_pred = np.multiply(y_pred, 18)
y_test = np.multiply(y_test, 18)

# Print different error metrics
rmse = RMSE()
bayer = Bayer()
kovatchev = Kovatchev()

print("RMSE: ", rmse(y_test, y_pred))
print("Bayer: ", bayer(y_test, y_pred))
print("Kovatchev: ", kovatchev(y_test, y_pred))

