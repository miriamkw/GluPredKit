from pyloopkit.exponential_insulin_model import percent_effect_remaining
from data_science_tidepool_api_python.makedata.tidepool_api import TidepoolAPI


EMAIL = 'YOUR_TIDEPOOL_USERNAME'
PASSWORD = 'YOUR_TIDEPOOL_PASSWORD'

tp_api = TidepoolAPI(EMAIL, PASSWORD)
tp_api.login()

# Default use the data from the last 24 hours
start_date = datetime.now() - timedelta(days=1)
end_date = datetime.now()

# Uncomment the lines below to customize days
#start_date = datetime(2023, 2, 4)
#end_date = datetime(2023, 2, 5) # year, month, day

# All the data in json format
user_data = tp_api.get_user_event_data(start_date, end_date)

tp_api.logout()



print(percent_effect_remaining(20, 360, 75))











