import pandas as pd
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
import numpy as np


def add_time_lagged_features(df, lagged_cols, num_lagged_features):
    lagged_df = df.copy()
    indexes = list(range(1, num_lagged_features + 1))

    for col in lagged_cols:
        for i in indexes:
            new_col_name = col + "_" + str(i * 5)
            lagged_df = pd.concat([lagged_df, lagged_df[col].shift(i).rename(new_col_name)], axis=1)
    return lagged_df


def add_what_if_features(df, what_if_cols, prediction_horizons):
    what_if_df = df.copy()
    prediction_index = prediction_horizons[0] // 5
    indexes = list(range(1, prediction_index + 1))

    for col in what_if_cols:
        for i in indexes:
            new_col_name = col + "_what_if_" + str(i * 5)
            what_if_df = pd.concat([what_if_df, what_if_df[col].shift(-i).rename(new_col_name)], axis=1)
    return what_if_df


def process_data(df, model_config_manager: ModelConfigurationManager, real_time=False):
    if "imputed" in df.columns:
        # Temporarily convert 'imputed' to float
        df['imputed'] = df['imputed'].astype(float)
        # Set entire rows to NaN where 'imputed' is True
        df.loc[df['imputed'] == 1.0, :] = np.nan
        df = df.drop(columns=['imputed'])

    subject_ids = df['id'].unique()
    subject_ids = list(filter(lambda x: not np.isnan(x), subject_ids))

    for subject_id in subject_ids:
        # Filter DataFrame for the current subject ID
        filter_df = df['id'] == subject_id

        # Add time-lagged features
        lagged_df = add_time_lagged_features(df.loc[filter_df, :], model_config_manager.get_num_features(),
                                             model_config_manager.get_num_lagged_features())

        # Join the DataFrames on their index columns and keep only the columns that are unique to each DataFrame
        df = df.join(lagged_df.drop(columns=df.columns), how='outer')
        filter_df = df['id'] == subject_id

        # Add what-if features
        what_if_df = add_what_if_features(df.loc[filter_df, :], model_config_manager.get_what_if_features(),
                                          model_config_manager.get_prediction_horizons())

        # Join the DataFrames on their index columns and keep only the columns that are unique to each DataFrame
        df = df.join(what_if_df.drop(columns=df.columns), how='outer')

    if real_time:
        df = df.dropna(subset=df.columns.difference(['target']))
    else:
        df = df.dropna()

    return df
