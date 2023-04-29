from datetime import datetime
import json
from src.parsers.tidepool_parser import fetch_tidepool_data
from src.models.loop_model import LoopModel

# Fetch training data from Tidepool API
start_date = datetime(2023, 3, 1)
end_date = datetime(2023, 3, 1)

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']

df_glucose, df_bolus, df_basal, df_carbs = fetch_tidepool_data(username, password, start_date, end_date)

# Define the output offset of 30 minutes into the future
output_offsets=[30]

# Train linear regression model
model = LoopModel()
model.predict(df_glucose, df_bolus, df_basal, df_carbs)


