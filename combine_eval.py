import os
import pandas as pd

# Path to the folder containing CSV files
folder_path = r'C:\Users\DoyoungOh\GluPredKit\data\reports\miriam'

# List to store all dataframes
dfs = []

# Loop through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        # Read the CSV file into a dataframe
        df = pd.read_csv(os.path.join(folder_path, file))
        # Append the dataframe to the list
        dfs.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Path to the output combined CSV file
output_file_path = r'C:\Users\DoyoungOh\GluPredKit\data\combined_reports.csv'

# Write the combined dataframe to a CSV file
combined_df.to_csv(output_file_path, index=False)

print("Combined CSV file saved successfully.")
