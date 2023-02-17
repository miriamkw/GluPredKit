from data_science_tidepool_api_python.makedata.tidepool_api import TidepoolAPI
from pyloopkit.generate_graphs import plot_loop_inspired_glucose_graph
from pyloopkit.loop_math import predict_glucose
from loop_model_scoring.penalty_math import (
    get_ideal_treatment,
    get_glucose_penalties_for_pairs,
    get_glucose_penalties
)
import json
from loop_model_scoring.tidepool_parser import (
    get_glucose_data,
    sort_by_first_list,
    remove_too_new_values,
    remove_too_old_values,
    get_offset,
	parse_json,
	parse_report_and_run
)
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


# Define start- and end-date for computation of penalties
start_date = datetime(2023, 2, 9, 12, 0)
end_date = datetime(2023, 2, 10, 16, 0)


# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)

username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']

tp_api = TidepoolAPI(username, password)
tp_api.login()

# All the data in json format
user_data = tp_api.get_user_event_data(start_date - timedelta(days=1), end_date + timedelta(days=1))

tp_api.logout()


# Load therapy settings
settings_file = "therapy_settings.json"
with open(settings_file, "r") as file:
    settings_dict = json.load(file)

# Get loop glucose predictions
(glucose_data, bolus_data, basal_data, carb_data) = parse_json(user_data)

offset = get_offset()

# Sort and filter out too new and too old glucose values
(glucose_dates, glucose_values) = get_glucose_data(glucose_data, offset=offset)
(glucose_dates, glucose_values) = sort_by_first_list(glucose_dates, glucose_values)[0:2]
(glucose_dates, glucose_values) = remove_too_old_values(start_date, glucose_dates, glucose_values)[0:2]
(glucose_dates, glucose_values) = remove_too_new_values(end_date + timedelta(hours=6), glucose_dates, glucose_values)[0:2]


# Plot measurements and dates
fig, ax = plt.subplots()
ax.scatter(glucose_dates[:-72], glucose_values[:-72], color='blue', label='True')
ax.set(xlabel='Datetime', ylabel='Blood Glucose (mg/dL)',
       title='Measured blood glucose values')
ax.grid()
plt.legend(loc='best')
plt.show()


# Plot penalties and dates
penalty_dates = []
penalties = []
n = len(glucose_dates) - 72

# Stop measuring for the last six hours
for i in range(0, n):
	# If the timedelta between a glucose measurement and a measurement 73 steps later is more than six hours (adding 10 minutes as margin)
	# countinue to next iteration as it means that measurements are missing in this time period
	if (glucose_dates[i+72] - glucose_dates[i]) > timedelta(hours=6, minutes=10):
		continue

	true_values = glucose_values[i:i+73]
	
	# Predict
	recommendations = parse_report_and_run(glucose_data, bolus_data, basal_data, carb_data, settings_dict, time_to_run=glucose_dates[i])
	derived_values = recommendations.get("predicted_glucose_values")[:73]

	penalty = np.mean(get_glucose_penalties_for_pairs(true_values, derived_values, penalty_type='error'))
	penalties.append(penalty)
	penalty_dates.append(glucose_dates[i])

fig, ax = plt.subplots()
ax.scatter(penalty_dates, penalties, color='blue', label='Penalty')
plt.axhline(y = np.mean(penalties), color='g', linestyle = '--')
ax.set(xlabel='Datetime', ylabel='Penalty',
       title='Penalties with respect to a forecast trajectory given a reference date')
ax.grid()
plt.legend(loc='best')
plt.show()




