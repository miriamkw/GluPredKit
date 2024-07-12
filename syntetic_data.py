import pandas as pd
import numpy as np

# Generating synthetic data
np.random.seed(0)  # For reproducibility

# Define number of rows (sample size)
num_rows = 6000

# Generate random data for each column
id_value = 1  # Single ID value
timestamps = pd.date_range('2024-07-01', periods=num_rows, freq='5min')
CGM_values = np.random.normal(130, 80, size=num_rows)  # Random CGM values between 70 and 180
insulin_values = np.random.uniform(0.0, 6.0, size=num_rows)  # Random insulin values between 0 and 2
carbs_values = np.random.uniform(0, 150, size=num_rows)  # Random carb intake between 0 and 50 grams
is_test = np.array([False] * (num_rows // 2) + [True] * (num_rows - (num_rows // 2)))

# Create dataframe
data = {
    'date': timestamps,
    'id': [id_value] * num_rows,
    'CGM': np.abs(CGM_values),
    'insulin': insulin_values,
    'carbs': carbs_values,
    'is_test': is_test
}

df = pd.DataFrame(data)
df.set_index('date')

# Print first few rows to verify
print(df.head())

# Store dataframe as CSV file
csv_filename = 'synthetic_data_with_datetime.csv'
df.to_csv(csv_filename)

print(f'\nDataFrame has been successfully saved as "{csv_filename}"')
