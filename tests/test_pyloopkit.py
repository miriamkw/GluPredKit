"""
import pandas as pd
import json
from glupredkit.models.loop import Model
from pyloopkit.loop_data_manager import update
from pyloopkit.dose import DoseType

# Parse json file
test_file = 'data/raw/live_capture_input.json'
f = open(test_file)
data = json.load(f)
f.close()

glucose_data = data['glucoseHistory']
df_glucose = pd.json_normalize(glucose_data)[['date', 'value']]
df_glucose.date = pd.to_datetime(df_glucose.date)
df_glucose.rename(columns={"value": "CGM"}, inplace=True)
df_glucose.sort_values(by='date', inplace=True, ascending=True)
df_glucose.set_index('date', inplace=True)

carb_data = data['carbEntries']
df_carbs = pd.json_normalize(carb_data)[['date', 'grams']]
df_carbs.date = pd.to_datetime(df_carbs.date)
df_carbs.rename(columns={"grams": "carbs"}, inplace=True)
df_carbs.sort_values(by='date', inplace=True, ascending=True)
df_carbs.set_index('date', inplace=True)

bolus_data = [dose for dose in data['doses'] if dose['type'] == 'bolus']
df_bolus = pd.json_normalize(bolus_data)[['startDate', 'volume']]
df_bolus.startDate = pd.to_datetime(df_bolus.startDate)
df_bolus.rename(columns={"startDate": "date", "volume": "bolus"}, inplace=True)
df_bolus.sort_values(by='date', inplace=True, ascending=True)
df_bolus.set_index('date', inplace=True)

basal_data = [dose for dose in data['doses'] if dose['type'] == 'basal']
df_basal = pd.json_normalize(basal_data)[['startDate', 'volume']]
df_basal.startDate = pd.to_datetime(df_basal.startDate)
df_basal.rename(columns={"startDate": "date", "volume": "basal"}, inplace=True)
df_basal.sort_values(by='date', inplace=True, ascending=True)
df_basal.set_index('date', inplace=True)

# Resampling all datatypes into the same time-grid
df = df_glucose.copy()
df = df.resample('5min', label='right').mean()

df_carbs = df_carbs.resample('5min', label='right').sum()
df = pd.merge(df, df_carbs, on="date", how='outer')
df['carbs'] = df['carbs']

df_bolus = df_bolus.resample('5min', label='right').sum()
df = pd.merge(df, df_bolus, on="date", how='outer')

df_basal = df_basal.resample('5min', label='right').sum()
df_basal['basal'] = df_basal['basal']
df = pd.merge(df, df_basal, on="date", how='outer')
df['basal'] = df['basal'].ffill(limit=12 * 24 * 2)
df['basal'] = df['basal'].fillna(value=0.0)

# Therapy settings
basal = data['basal'][0]['value']
carb_ratio = data['carbRatio'][0]['value']
insulin_sensitivity = data['sensitivity'][0]['value']

print(f'Therapy settings basal: {basal} carb: {carb_ratio} isf: {insulin_sensitivity}')

# Make prediction and print
loop_model = Model(prediction_horizon=180)
input_dict = loop_model.get_input_dict(insulin_sensitivity, carb_ratio, basal)
input_dict["time_to_calculate_at"] = df.iloc[-1].name

input_dict["glucose_dates"] = list(df[df['CGM'].notna()].index)
input_dict["glucose_values"] = list(df[df['CGM'].notna()].CGM)

input_dict["carb_dates"] = list(df[df['carbs'] > 0].index)
input_dict["carb_values"] = list(df[df['carbs'] > 0].carbs)
input_dict["carb_absorption_times"] = [180 for _ in df[df['carbs'] > 0].carbs]

# TODO: Basal or tempbasal here?
basal_dose_types = [DoseType.from_str("tempbasal") for _ in df[df['basal'].notna()].basal]
basal_start_times = df[df['basal'].notna()].index
basal_end_times = [val + pd.Timedelta(minutes=5) for val in df[df['basal'].notna()].index]
basal_values = df[df['basal'].notna()].basal
basal_units = [None for _ in df[df['basal'].notna()].basal]

bolus_dose_types = [DoseType.from_str("bolus") for _ in df[df['bolus'].notna()].bolus]
bolus_start_times = df[df['bolus'].notna()].index
bolus_end_times = df[df['bolus'].notna()].index
bolus_values = df[df['bolus'].notna()].bolus
bolus_units = [None for _ in df[df['bolus'].notna()].bolus]

# Step 1: Combine into tuples
combined_basal = list(zip(basal_dose_types, basal_start_times, basal_end_times, basal_values, basal_units))
combined_bolus = list(zip(bolus_dose_types, bolus_start_times, bolus_end_times, bolus_values, bolus_units))

# Step 2: Merge lists
combined = combined_basal + combined_bolus

# Step 3: Sort by the start times (second element of each tuple)
combined.sort(key=lambda x: x[1])

# Step 4: Separate into individual lists
dose_types, start_times, end_times, values, units = zip(*combined)

input_dict["dose_types"] = dose_types
input_dict["dose_start_times"] = start_times
input_dict["dose_end_times"] = end_times
input_dict["dose_values"] = values
input_dict["dose_delivered_units"] = units

print(input_dict["time_to_calculate_at"])

output_dict = update(input_dict)
prediction_result = output_dict.get("predicted_glucose_values")
prediction_dates = output_dict.get("predicted_glucose_dates")

for i in range(len(prediction_dates)):
    print(f'Pred: {prediction_result[i]} Date: {prediction_dates[i]}')
"""



