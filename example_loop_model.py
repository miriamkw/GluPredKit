from datetime import datetime
import json
from src.parsers.tidepool_parser import TidepoolParser
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

tidepool_parser = TidepoolParser()
df_glucose, df_bolus, df_basal, df_carbs = tidepool_parser(start_date, end_date, username, password)

output_index = 5
model = LoopModel()
y_pred = model.predict(df_glucose, df_bolus, df_basal, df_carbs)
y_pred = [predictions[output_index] for predictions in y_pred]
# The 6th value is the first reference value because of the history data in predictions
_, y_true = model.get_glucose_data(df_glucose)
y_true = y_true[6 + output_index:len(y_pred) + 6 + output_index]


# Print different error metrics
rmse = RMSE()
bayer = Bayer()
kovatchev = Kovatchev()

print("Error metrics after " + str((output_index + 1)*5) + " minutes:")
print("RMSE: ", rmse(y_true, y_pred))
print("Bayer: ", bayer(y_true, y_pred))
print("Kovatchev: ", kovatchev(y_true, y_pred))
