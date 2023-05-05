from datetime import datetime
import json
from src.parsers.tidepool_parser import TidepoolParser
from src.models.loop_model import LoopModel
from src.models.linear_regressor import LinearRegressor
from src.plots.prediction_trajectories import PredictionTrajectories

# Fetch training data from Tidepool API
start_date = datetime(2023, 5, 4)
end_date = datetime(2023, 5, 4)

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']

tidepool_parser = TidepoolParser()
df_glucose, df_bolus, df_basal, df_carbs = tidepool_parser(start_date, end_date, username, password)

output_offset = None
#model = LoopModel()
model = LinearRegressor()
y_pred, y_true = model.predict(df_glucose, df_bolus, df_basal, df_carbs, output_offset=output_offset)

plot = PredictionTrajectories()
plot(y_true, y_pred)
