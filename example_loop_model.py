from datetime import datetime
import json
from src.parsers.tidepool_parser import fetch_tidepool_data
from src.models.loop_model import LoopModel
from src.metrics.rmse import RMSE
from src.metrics.bayer import Bayer
from src.metrics.kovatchev import Kovatchev

# Fetch training data from Tidepool API
start_date = datetime(2023, 5, 3)
end_date = datetime(2023, 5, 3)

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']

df_glucose, df_bolus, df_basal, df_carbs = fetch_tidepool_data(username, password, start_date, end_date)

output_offset = 30
model = LoopModel()
y_pred, y_true = model.predict(df_glucose, df_bolus, df_basal, df_carbs, output_offset=output_offset)

# Print different error metrics
rmse = RMSE()
bayer = Bayer()
kovatchev = Kovatchev()

print("Error metrics after " + str(output_offset) + " minutes:")
print("RMSE: ", rmse(y_true, y_pred))
print("Bayer: ", bayer(y_true, y_pred))
print("Kovatchev: ", kovatchev(y_true, y_pred))
