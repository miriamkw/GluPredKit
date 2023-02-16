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
    remove_too_old_values,
    get_offset,
	parse_json,
	parse_report_and_run
)
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

"""
This example computes a forecast for a given date (time_to_run) and the penalty for this forecasted trajectory.

"""

# The time to run a prediction
# Assuming that there are available continous measurements after in the Tidepool API
# Finds the last glucose value before this date
time_to_run = datetime(2023, 2, 5, 8, 25)


# Load data from Tidepool API
with open('credentials.json', 'r') as f:
    credentials = json.load(f)


username = credentials['tidepool_api']['email']
password = credentials['tidepool_api']['password']


tp_api = TidepoolAPI(username, password)
tp_api.login()


# Specify the dates for fetching data (the API only accepts dates, so specifying time will be ignored)
start_date = time_to_run - timedelta(days=1)
end_date = time_to_run + timedelta(days=1)

# All the data in json format
user_data = tp_api.get_user_event_data(start_date, end_date)

tp_api.logout()


# Load therapy settings
settings_file = "therapy_settings.json"
with open(settings_file, "r") as file:
    settings_dict = json.load(file)

(glucose_data, bolus_data, basal_data, carb_data) = parse_json(user_data)

offset = get_offset()
(dates, measurements) = get_glucose_data(glucose_data, offset=offset)
(dates, measurements) = sort_by_first_list(dates, measurements)[0:2]
(dates, measurements) = remove_too_old_values(time_to_run, dates, measurements)[0:2]


recommendations = parse_report_and_run(glucose_data, bolus_data, basal_data, carb_data, settings_dict, time_to_run=time_to_run)
inputs = recommendations.get("input_data")

(insulin_predicted_glucose_dates,
 insulin_predicted_glucose_values
 ) = predict_glucose(
     time_to_run, inputs.get("glucose_values")[-1],
     insulin_effect_dates=recommendations.get("insulin_effect_dates"),
     insulin_effect_values=recommendations.get("insulin_effect_values")
     )

(carb_predicted_glucose_dates,
 carb_predicted_glucose_values
 ) = predict_glucose(
     time_to_run, inputs.get("glucose_values")[-1],
     carb_effect_dates=recommendations.get("carb_effect_dates"),
     carb_effect_values=recommendations.get("carb_effect_values")
     )


# Plot predicted vs measured
t = np.arange(-5.0*11, 365.0, 5.0)

fig, ax = plt.subplots()
ax.scatter(t[:12+73], inputs.get("glucose_values")[-12:] + measurements[:72], color='blue', label='True')
ax.plot(t[-73:], recommendations.get("predicted_glucose_values")[:73], label='Predicted', linestyle = '--', color='orange')
ax.plot(t[-71:-7], carb_predicted_glucose_values[:73], label='Carbs', linestyle = '--')
ax.plot(t[-73:], insulin_predicted_glucose_values[:73], label='Insulin', linestyle = '--')

ax.set(xlabel='Time (minutes)', ylabel='Blood Glucose (mg/dL)',
       title='Measured vs one trajectory of predicted values')
ax.grid()
plt.legend(loc='best')

plt.show()


# Plot the treatment decisions made using derived vs true values
#reference_blood_glucose = inputs.get("glucose_values")[-1] # 95
true_blood_glucose = measurements[71] # 174
derived_blood_glucose = recommendations.get("predicted_glucose_values")[72] #112
target = 105

ideal_glucose_values = get_ideal_treatment(true_blood_glucose, target)
derived_end_value = true_blood_glucose + target - derived_blood_glucose
derived_glucose_values = get_ideal_treatment(true_blood_glucose, derived_end_value)

derived_penalties = get_glucose_penalties(derived_glucose_values)
true_penalties = get_glucose_penalties(ideal_glucose_values)

t = np.arange(0.0, 361.0, 1.0)
fig, ax = plt.subplots()
ax.plot(t, derived_glucose_values, label='Derived')
ax.plot(t, ideal_glucose_values, label='True')

ax.set(xlabel='Time (minutes)', ylabel='Simulated Blood Glucose (mg/dL)',
       title='Treatment Decisions for prediction when t=360')
ax.grid()
plt.legend(loc='best')

plt.show()


# Plot the penalty of simulated true blood glucose traces
derived_penalties = get_glucose_penalties(derived_glucose_values)
true_penalties = get_glucose_penalties(ideal_glucose_values)

fig, ax = plt.subplots()
plt.axhline(y = np.mean(derived_penalties), color='g', linestyle = '--')
plt.axhline(y = np.mean(true_penalties), color='g', linestyle = '--')
ax.plot(t, derived_penalties, label='Derived')
ax.plot(t, true_penalties, label='True')

ax.set(xlabel='Time (minutes)', ylabel='Penalty',
       title='Penalty of Simulated True Blood Glucose Traces for prediction when t=360')
ax.grid()
plt.legend(loc='best')

plt.show()


# Plot predicted vs measured vs penalties
t = np.arange(-5.0*11, 365.0, 5.0)
true_values = [inputs.get("glucose_values")[-1]] + measurements[:72]
derived_values = recommendations.get("predicted_glucose_values")[:73]
penalties = get_glucose_penalties_for_pairs(true_values, derived_values, penalty_type='rmse')

fig, ax = plt.subplots()
ax.scatter(t[:12+73], inputs.get("glucose_values")[-12:] + measurements[:72], color='blue', label='True')
ax.plot(t[-73:], recommendations.get("predicted_glucose_values")[:73], label='Predicted', linestyle = '--', color='orange')
ax.plot(t[-73:], penalties, label='Penalty', linestyle = '--', color='red')

ax.set(xlabel='Time (minutes)', ylabel='Blood Glucose (mg/dL)',
       title='Measured vs one trajectory of predicted values')
ax.grid()
plt.legend(loc='best')

plt.show()



