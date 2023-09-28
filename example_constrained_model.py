import datetime
import json
from src.parsers.tidepool_parser import TidepoolParser
from src.parsers.oura_api_parser import OuraRingParser
from src.processors.resampler import Resampler
from src.models.loop_model import LoopModel
from src.metrics.rmse import RMSE
from src.metrics.bayer import Bayer
from src.metrics.kovatchev import Kovatchev

# Fetch training data from Tidepool API
#start_date = datetime.datetime(2023, 7, 1) # yyyy mm dd
end_date = datetime.datetime(2023, 7, 26)
#end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=5)

# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']
access_token = credentials['oura_api']['access_token']

tidepool_parser = TidepoolParser()
df_glucose, df_bolus, df_basal, df_carbs, df_workouts = tidepool_parser(start_date, end_date, username, password)

oura_parser = OuraRingParser()
df_oura = oura_parser(start_date, end_date, access_token)

resampler = Resampler()
#resampler(df_glucose, df_bolus, df_basal, df_carbs, df_workouts, df_oura)

#end_date = None

# Datetime at the start of eating a huge ice cream
datetime_ice_cream = datetime.datetime(2023, 7, 25, 11, 10, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))

datetime_insulin = datetime.datetime(2023, 7, 25, 12, 35, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))

datetime_hypo = datetime.datetime(2023, 7, 25, 16, 00, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))

datetime_banana = datetime.datetime(2023, 7, 25, 17, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))

datetime_end = datetime.datetime(2023, 7, 25, 19, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))

print(datetime_banana)

resampler.get_measurements(df_glucose, start_date=datetime_banana, n_samples=12*3, use_mgdl=False)

df = resampler.get_most_recent(df_glucose, df_bolus, df_basal, df_carbs, df_workouts, df_oura,
                               basal_rate=0.7,
                               end_date=datetime_banana)

#print(df)
