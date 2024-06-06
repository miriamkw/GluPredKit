"""
The Open APS parser is processing the .pkl file produced by using the data cleaner by Harry Emerson: https://github.com/hemerson1/OpenAPS_Cleaner.
"""
import pandas as pd
import pickle
from .base_parser import BaseParser


class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, file_path: str, *args):
        """
        file_path -- the file path to the processed data including <file_name>.pkl.
        """
        df = pd.read_csv(file_path, low_memory=False)
        relevant_columns = ['date', 'bg', 'basal', 'carbs', 'bolus', 'PtID']
        df = df[relevant_columns]
        df = df.rename(columns={'bg': 'CGM', 'PtID': 'id'})
        df['date'] = pd.to_datetime(df['date'])

        # Sort by 'id' and the index (date)
        df = df.sort_values(by=['id', 'date'])

        # Set date to index
        df = df.set_index('date')

        # Function to validate the time intervals
        def validate_intervals(group):
            # Calculate the time difference between consecutive dates
            time_diff = group.index.to_series().diff().dt.total_seconds().dropna()
            # Check if all time differences are exactly 300 seconds (5 minutes)
            valid = (time_diff == 300).all()
            if not valid:
                print(f"ID {group['id'].iloc[0]} has invalid intervals.")
            return valid

        # Group by 'id' and apply the validation function
        valid_intervals = df.groupby('id').apply(validate_intervals)

        if valid_intervals.all():
            print("All IDs have valid 5-minute intervals with no bigger breaks than 5 minutes.")
        else:
            print("There are IDs with invalid intervals.")

        return df

