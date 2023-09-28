import pandas as pd
import numpy as np
import datetime
import re


def add_activity_states(start_date, duration, workout_name, df):
    end_date = start_date + pd.to_timedelta(duration, unit='s')
    df['activity_state'][df.index.to_series().between(start_date, end_date)] = workout_name


def add_time_lagged_features(col_name, df, samples=72):
    indexes = list(range(1, samples + 1))

    for i in indexes:
        new_col_name = col_name + "_" + str(i * 5)
        df = pd.concat([df, df[col_name].shift(i).rename(new_col_name)], axis=1)
    return df


def add_what_if_features(col_name, df, samples=36, value=None, interval=5):
    """
    col_name -- name of the column that you are adding what if events for
    df -- dataframe
    samples -- number of samples of what if events
    value -- predefined value, if you want to use the same predefined value for all the what-if events.
    interval -- the interval between each added what-if event in minutes
    """
    n_intervals = int(interval / 5)  # Intervals from minutes to number of elements

    indexes = list(range(n_intervals, samples + 1, n_intervals))

    for i in indexes:
        new_col_name = col_name + "_what_if_" + str(i * 5)

        if value == None:
            df = pd.concat([df, df[col_name].shift(-i).rename(new_col_name)], axis=1)
        else:
            new_column_df = pd.DataFrame({new_col_name: [value] * len(df)}, index=df.index)
            df = pd.concat([df, new_column_df], axis=1)
    return df


def map_activity_to_number(df):
    # Changing activity state to numeric value
    activity_cols = [el for el in df.columns if el.startswith("activity_state")]

    for col in activity_cols:
        df[col] = df[col].replace({
            'None': 1,
            'Walking': 5,
            'Running': 10,
            'Cycling': 7,
            'TraditionalStrengthTraining': 3,
            'Swimming': 7,
            'Pilates': 2,
            'MixedCardio': 8,
            'MindAndBody': 2,
            'Yoga': 3,
            'Elliptical': 7,
            'CrossCountrySkiing': 7,
            'Other': 1,
            'Sleep': 0,
        })

    return df


def aggregate_hours(df, interval):
    # Using a pre-defined hour interval for hour of day
    for i in range(1, 24):
        df['hour'] = df['hour'].replace(i, int(i / interval))

    return df


def filter_for_output_offset(output_offset, df):
    """
    Get the relevant target for the output offset
    Filter out the irrelevant what if events for the output offset
    """
    target_col = 'target_' + str(output_offset)
    y = df[target_col]

    # Drop target columns and irrelevant what if columns (that are after output offset minutes ahead)
    target_cols = [el for el in df.columns if el.startswith("target")]

    if output_offset < 100:
        offset_string = str(output_offset)
        pattern = r'what_if_([' + offset_string[0] + '-9][' + offset_string[1] + '-9]|\d{3,})$'
        drop_cols = target_cols + [col for col in df.columns if re.search(pattern, col)]
    else:
        offset_string = str(output_offset)
        pattern = r'what_if_([' + offset_string[0] + '-9][' + offset_string[1] + '-9][' + offset_string[2] + '-9])$'
        drop_cols = target_cols + [col for col in df.columns if re.search(pattern, col)]

    # Drop time-lagged events earlier than 260 minutes before the time of prediction
    threshold_number_cgm = 260 - output_offset
    threshold_number_activity_state = output_offset - 60
    for col in df.columns:
        if col.startswith("carbs_") or col.startswith("insulin_") or col.startswith("CGM_"):
            if len(col.split("_")) == 2:
                if int(col.split("_")[1]) >= threshold_number_cgm:
                    drop_cols = drop_cols + [col]
        # Drop activity state more than 30 minutes before the time of prediction
        if col.startswith("activity_state") and threshold_number_activity_state > 0:
            if len(col.split("_")) <= 2:
                drop_cols = drop_cols + [col]
            elif int(col.split("_")[4]) < threshold_number_activity_state:
                drop_cols = drop_cols + [col]

    X = df.copy().drop(columns=drop_cols)

    return X, y


def resample(df_glucose, df_bolus, df_basal, df_carbs, df_workouts, df_oura=None):

    df_glucose.drop(columns=(["units", "device_name"]), inplace=True)
    df_glucose.rename(columns={"value": "CGM", "time": "date"}, inplace=True)
    df_glucose.sort_values(by='date', inplace=True, ascending=True)
    df_glucose.set_index('date', inplace=True)

    df_bolus.drop(columns=(["device_name"]), inplace=True)
    df_bolus.rename(columns={"dose[IU]": "insulin", "time": "date"}, inplace=True)
    df_bolus.sort_values(by='date', inplace=True, ascending=True)
    df_bolus.set_index('date', inplace=True)

    df_basal.drop(columns=(["device_name", "duration[ms]", "scheduled_basal", "delivery_type"]), inplace=True)
    df_basal.rename(columns={"duration[ms]": "duration", "rate[U/hr]": "basal_rate", "time": "date"}, inplace=True)
    df_basal.sort_values(by='date', inplace=True, ascending=True)
    df_basal.set_index('date', inplace=True)

    df_carbs.drop(columns=(["units", "absorption_time[s]"]), inplace=True)
    df_carbs.rename(columns={"value": "carbs", "time": "date"}, inplace=True)
    df_carbs.sort_values(by='date', inplace=True, ascending=True)
    df_carbs.set_index('date', inplace=True)

    # Resampling all datatypes into the same time-grid
    df = df_glucose.copy()
    df = df.resample('5T', label='right').mean()

    df_carbs = df_carbs.resample('5T', label='right').sum().fillna(value=0)
    df = pd.merge(df, df_carbs, on="date", how='outer')
    df['carbs'] = df['carbs'].fillna(value=0.0)

    df_bolus = df_bolus.resample('5T', label='right').sum()
    df = pd.merge(df, df_bolus, on="date", how='outer')

    df_basal = df_basal.resample('5T', label='right').last()
    df_basal['basal_rate'] = df_basal['basal_rate'] / 60 * 5  # From U/hr to U (5-minutes)
    df = pd.merge(df, df_basal, on="date", how='outer')
    df['basal_rate'] = df['basal_rate'].ffill(limit=12 * 24 * 2)
    df[['insulin', 'basal_rate']] = df[['insulin', 'basal_rate']].fillna(value=0.0)
    df['insulin'] = df['insulin'] + df['basal_rate']
    df.drop(columns=(["basal_rate"]), inplace=True)

    # Add activity states
    df['activity_state'] = "None"
    df_workouts.apply(lambda x: add_activity_states(x['time'], x['duration[s]'], x['name'], df), axis=1)

    # Get the current datetime in UTC, given the calendar on current computer
    current_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    df.index = df.index.tz_convert(current_timezone)

    if not df_oura is None:
        df_oura['day'] = pd.to_datetime(df_oura['day']).dt.tz_localize(current_timezone) + pd.to_timedelta(7,
                                                                                                           unit='h')
        df_oura.rename(columns={"day": "date"}, inplace=True)
        df_oura.sort_values(by='date', inplace=True, ascending=True)
        df_oura.set_index('date', inplace=True)
        df_oura = df_oura.resample('5T', label='right').mean()  # .ffill(limit=12*24*2)
        df = pd.merge(df, df_oura, on="date", how='outer')

        # Imputation
        df[df_oura.columns] = df[df_oura.columns].ffill(limit=12 * 24 * 2)

    return df


class Resampler():
    def __init__(self):
        super().__init__

    def __call__(self, df_glucose, df_bolus, df_basal, df_carbs, df_workouts, df_oura=None):
        time_lagged_samples = 72
        what_if_samples = 36

        df = resample(df_glucose, df_bolus, df_basal, df_carbs, df_workouts, df_oura)

        # Imputation
        df['CGM'] = df.CGM.ffill(limit=1)

        # Feature addition
        df['hour'] = df.index.copy().to_series().apply(lambda x: x.hour)

        time_lagged_cols = ['CGM', 'insulin', 'carbs']
        for col in time_lagged_cols:
            df = add_time_lagged_features(col, df, samples=time_lagged_samples)

        what_if_cols = ['insulin', 'carbs', 'activity_state']
        for col in what_if_cols:
            df = add_what_if_features(col, df, samples=what_if_samples)

        df = add_what_if_features('CGM', df, samples=what_if_samples, interval=30)

        # Add targets
        target_indexes = list(range(6, 37, 6))
        for i in target_indexes:
            col_name = 'target_' + str(int(i * 5))
            df = pd.concat([df, df.CGM.shift(-i).rename(col_name)], axis=1)

        df = df.copy()
        df.reset_index(drop=True, inplace=True)
        df['CGM_avg_12h'] = float('nan')
        df['CGM_12h_hypo_count'] = float('nan')

        for i, row in df.iterrows():
            # Calculate the start and end indices for the 12-hour window
            start_index = max(0, i - 144)
            end_index = i + 1

            # Extract the values within the 12-hour window
            window_values = df.iloc[start_index:end_index]['CGM']

            # Calculate the average and assign it to the 'avg_12h' column
            df.at[i, 'CGM_avg_12h'] = np.nanmean(window_values)

            # Count the hypos and assign it to the 'CGM_12h_hypo_count' column
            df.at[i, 'CGM_12h_hypo_count'] = window_values[window_values < 72].shape[0]

        # Aggregate hour into 2-hour intervals
        df = aggregate_hours(df, interval=2)
        df = map_activity_to_number(df)

        df = df.dropna()

        # Train and test split
        split_index = int(len(df) * 0.8)
        margin = 12 * 24

        # Split the data into train and test sets
        train_data = df[:split_index - margin]
        test_data = df[split_index + margin:]

        # Storing dataframe in a folder
        train_data.to_csv('data/train.csv')
        test_data.to_csv('data/test.csv')

    def get_most_recent(self, df_glucose, df_bolus, df_basal, df_carbs, df_workouts, df_oura=None, basal_rate=0.7, end_date=None):
        time_lagged_samples = 72
        what_if_samples = 36

        df = resample(df_glucose, df_bolus, df_basal, df_carbs, df_workouts, df_oura)

        # Imputation
        df['CGM'] = df.CGM.ffill(limit=1)

        # Feature addition
        df['hour'] = df.index.copy().to_series().apply(lambda x: x.hour)

        time_lagged_cols = ['CGM', 'insulin', 'carbs']
        for col in time_lagged_cols:
            df = add_time_lagged_features(col, df, samples=time_lagged_samples)

        df = add_what_if_features('insulin', df, samples=what_if_samples, value=basal_rate/12)
        df = add_what_if_features('carbs', df, samples=what_if_samples, value=0)
        df = add_what_if_features('activity_state', df, samples=what_if_samples, value=1)
        df = add_what_if_features('CGM', df, samples=what_if_samples, value=0)
        """
        df = add_what_if_features('insulin', df, samples=what_if_samples)
        df = add_what_if_features('carbs', df, samples=what_if_samples)
        df = add_what_if_features('activity_state', df, samples=what_if_samples)
        df = add_what_if_features('CGM', df, samples=what_if_samples)
        """
        df = df.copy()

        if end_date:
            df = df[df.index <= end_date]

        df.reset_index(drop=True, inplace=True)
        df['CGM_avg_12h'] = float('nan')
        df['CGM_12h_hypo_count'] = float('nan')

        for i, row in df.iterrows():
            # Calculate the start and end indices for the 12-hour window
            start_index = max(0, i - 144)
            end_index = i + 1

            # Extract the values within the 12-hour window
            window_values = df.iloc[start_index:end_index]['CGM']

            # Calculate the average and assign it to the 'avg_12h' column
            df.at[i, 'CGM_avg_12h'] = np.nanmean(window_values)

            # Count the hypos and assign it to the 'CGM_12h_hypo_count' column
            df.at[i, 'CGM_12h_hypo_count'] = window_values[window_values < 72].shape[0]

        # Aggregate hour into 2-hour intervals
        df = aggregate_hours(df, interval=2)
        df = map_activity_to_number(df)

        df = df.dropna()

        # We only extract the last row to store
        df.to_csv('data/current.csv')
        return df


    def get_measurements(self, df_glucose, start_date, n_samples, use_mgdl=True):
        """
        Get n amount of CGM samples after a given start_date
        """
        if use_mgdl:
            k = 1
        else:
            k = 18.0182

        print(df_glucose)
        df_glucose = df_glucose[df_glucose.time >= start_date]
        df_glucose = df_glucose.sort_values(by='time', ascending=True)

        values = [el / k for el in df_glucose['value'][:n_samples]]
        print(values)

        return values




