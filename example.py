from loop_model_scoring.penalty_math import (
    get_ideal_treatment,
    get_average_glucose_penalty,
    get_glucose_penalties
)
#from datetime import datetime
#import json
import matplotlib.pyplot as plt
import numpy as np
#from data_science_tidepool_api_python.makedata.tidepool_api import TidepoolAPI

"""
Recreating the examples in the Loop Model Scoring documentation.
"""

"""
# TODO: Add support for using Tidepool API in this example

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
"""

reference_blood_glucose = 90
true_blood_glucose = 90
derived_blood_glucose = 140
target = 105

ideal_glucose_values = get_ideal_treatment(reference_blood_glucose, target)
derived_end_value = true_blood_glucose + target - derived_blood_glucose
derived_glucose_values = get_ideal_treatment(reference_blood_glucose, derived_end_value)


# Plot the treatment decisions made using derived vs true values
t = np.arange(0.0, 361.0, 1.0)
fig, ax = plt.subplots()
ax.plot(t, derived_glucose_values, label='Derived')
ax.plot(t, ideal_glucose_values, label='True')

ax.set(xlabel='Time (minutes)', ylabel='Simulated Blood Glucose (mg/dL)',
       title='Treatment Decisions')
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
       title='Penalty of Simulated True Blood Glucose Traces')
ax.grid()
plt.legend(loc='best')

plt.show()




# TODO (later):
# Create a file in this repo where you can use tidepool api to calcualte penalty for a given date and time
























