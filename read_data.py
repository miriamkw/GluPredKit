import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'data/raw/OhioT1DM.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

print(df)

# Get unique IDs from the column that contains the IDs
unique_ids = df['id'].unique()  # Replace 'id_column' with the actual column name containing IDs

# Iterate over each unique ID, filter the DataFrame, and print the info
for unique_id in unique_ids:
    filtered_df = df[df['id'] == unique_id]

    print(f"Info for ID: {unique_id}")
    print(filtered_df[(filtered_df['exercise'] != 0.0) & (filtered_df['exercise'].notna())].shape)

    # Separate train and test data
    print("Exercise train", filtered_df[(filtered_df['exercise'] != 0.0) & (filtered_df['exercise'].notna()) & (filtered_df['is_test'] == False)].shape)
    print("Exercise test", filtered_df[(filtered_df['exercise'] != 0.0) & (filtered_df['exercise'].notna()) & (filtered_df['is_test'])].shape)

    #print(filtered_df.info())
    print("-" * 50)  # Just a separator for readability


# Conclusion: ID 591, 588, _570_, 563 are best data



