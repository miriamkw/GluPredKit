from data_science_tidepool_api_python.makedata.tidepool_api import TidepoolAPI
from loop_model_scoring.penalty_math import (
    get_glucose_penalties_for_pairs
)
import json
from loop_model_scoring.tidepool_parser import (
    get_glucose_data,
    sort_by_first_list,
    remove_too_new_values,
    remove_too_old_values,
    get_offset,
    parse_json,
    #parse_report_and_run,
    parse_settings,
    parse_report,
    run_prediction
)
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


# Define start- and end-date for computation of penalties
start_date = datetime(2023, 2, 9, 12, 0)
end_date = datetime(2023, 2, 10, 16, 0)
penalty_type = 'bayer'

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

# offset = get_offset()

input_dict, glucose_dates, glucose_values, dose_types, dose_starts, dose_ends, dose_values, carb_dates, carb_values, carb_absorptions = parse_report(glucose_data, bolus_data, basal_data, carb_data)

# Sort and filter out too new and too old glucose values
(glucose_dates, glucose_values) = remove_too_old_values(start_date, glucose_dates, glucose_values)[0:2]
(glucose_dates, glucose_values) = remove_too_new_values(end_date + timedelta(hours=6), glucose_dates, glucose_values)[0:2]




n = len(glucose_dates) - 72



# Therapy settings to test
carb_ratios = np.linspace(8, 12, 5)
# Linear combination of basal and ISF
basal_sens_combinations = np.linspace(0, 4, 5)

"""
Insulin sensitivity above when 0 gives:
- basal = 1.0
- 4.5

Insulin sensitivity above when 4 gives:
- basal = 0.6
- ISF = 5.0
"""

# Carb ratios
X = []
# Basal sensitivity combination
Y = []
# Penalties
Z = []

for carb_ratio in carb_ratios:
    x = []
    y = []
    z = []
    for basal_sens in basal_sens_combinations:
        x.append(carb_ratio)
        y.append(basal_sens)
        
        settings_dict["carb_ratio_schedule"] = [
            {
                "startTime": 0.0, 
                "value": carb_ratio
            }
        ]
        settings_dict["insulin_sensitivity_factor_schedule"] = [
            {
                "startTime": 0.0, 
                "value": 4.5*18.0 + basal_sens*0.1*18.0
            }
        ]
        settings_dict["basal_rate_schedule"] = [
            {
                "startTime": 0.0, 
                "value": 1.0 - basal_sens*0.1
            }
        ]
        # Calculate the penalties for each prediction given the settings
        penalties = []
        for i in range(0, n):
            # If the timedelta between a glucose measurement and a measurement 73 steps later is more than six hours (adding 10 minutes as margin)
            # countinue to next iteration as it means that measurements are missing in this time period
            if (glucose_dates[i+72] - glucose_dates[i]) > timedelta(hours=6, minutes=10):
                continue

            true_values = glucose_values[i:i+73]
            
            # Predict
            # recommendations = parse_report_and_run(glucose_data, bolus_data, basal_data, carb_data, settings_dict, time_to_run=glucose_dates[i])
            input_dict = parse_settings(input_dict, settings_dict)
            recommendations = run_prediction(input_dict, glucose_dates.copy(), glucose_values.copy(), dose_types.copy(), dose_starts.copy(), dose_ends.copy(), dose_values.copy(), carb_dates.copy(), carb_values.copy(), carb_absorptions.copy(), glucose_dates[i])
            derived_values = recommendations.get("predicted_glucose_values")[:73]

            penalty = np.mean(get_glucose_penalties_for_pairs(true_values, derived_values, penalty_type=penalty_type))
            penalties.append(penalty)
        print('Carb ratio: ', carb_ratio)
        print('Insulin basal sens: ', basal_sens)
        print('Penalty: ', np.mean(penalties))

        z.append(np.mean(penalties))

    X.append(x)
    Y.append(y)
    Z.append(z)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

## 3D Plot of Penalties
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

plt.show()


# Therapy settings to test
basal_rates = np.linspace(0.8, 1.2, 5)
# Linear combination of basal and ISF
ins_sens_factors = np.linspace(4.5*18.0, 5.0*18.0, 5)

# We learned from the script above the optimal carb ratio by taking the one with the average lowest penalty 
carb_penalty_sums = [sum(inner_lst) for inner_lst in Z]
min_sum_index = carb_penalty_sums.index(min(carb_penalty_sums))
carb_ratio = carb_ratios[min_sum_index]

# Basal rates
X = []
# Insulin sensitivity
Y = []
# Penalties
Z = []

settings_dict["carb_ratio_schedule"] = [
            {
                "startTime": 0.0, 
                "value": carb_ratio
            }
        ]

for basal_rate in basal_rates:
    x = []
    y = []
    z = []
    for ins_sens in ins_sens_factors:
        x.append(basal_rate)
        y.append(ins_sens)
        
        settings_dict["insulin_sensitivity_factor_schedule"] = [
            {
                "startTime": 0.0, 
                "value": ins_sens
            }
        ]
        settings_dict["basal_rate_schedule"] = [
            {
                "startTime": 0.0, 
                "value": basal_rate
            }
        ]
        # Calculate the penalties for each prediction given the settings
        penalties = []
        for i in range(0, n):
            # If the timedelta between a glucose measurement and a measurement 73 steps later is more than six hours (adding 10 minutes as margin)
            # countinue to next iteration as it means that measurements are missing in this time period
            if (glucose_dates[i+72] - glucose_dates[i]) > timedelta(hours=6, minutes=10):
                continue

            true_values = glucose_values[i:i+73]
            
            # Predict
            # recommendations = parse_report_and_run(glucose_data, bolus_data, basal_data, carb_data, settings_dict, time_to_run=glucose_dates[i])
            input_dict = parse_settings(input_dict, settings_dict)
            recommendations = run_prediction(input_dict, glucose_dates.copy(), glucose_values.copy(), dose_types.copy(),
                                             dose_starts.copy(), dose_ends.copy(), dose_values.copy(),
                                             carb_dates.copy(), carb_values.copy(), carb_absorptions.copy(),
                                             glucose_dates[i])
            derived_values = recommendations.get("predicted_glucose_values")[:73]

            penalty = np.mean(get_glucose_penalties_for_pairs(true_values, derived_values, penalty_type=penalty_type))
            penalties.append(penalty)
        print('Basal rate: ', basal_rate)
        print('Insulin sensitivity: ', ins_sens)
        print('Penalty: ', np.mean(penalties))

        z.append(np.mean(penalties))

    X.append(x)
    Y.append(y)
    Z.append(z)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

## 3D Plot of Penalties
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

plt.show()


"""
TODO:
- Right now a bottleneck is runtime --> this needs to be improved
- Create a more sophisticated process with testing and validation datasets, maybe in a notebook
- Add a stage where you look at optimization for time on day

"""




