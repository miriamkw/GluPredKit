import pandas as pd
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
import numpy as np


def add_time_lagged_features(df, lagged_cols, num_lagged_features):
    lagged_df = pd.DataFrame()
    indexes = list(range(1, num_lagged_features + 1))

    for col in lagged_cols:
        for i in indexes:
            new_col_name = col + "_" + str(i * 5)
            lagged_df = pd.concat([lagged_df, df[col].shift(i).rename(new_col_name)], axis=1)
    return lagged_df


def add_what_if_features(df, what_if_cols, prediction_horizon):
    what_if_df = pd.DataFrame()
    prediction_index = prediction_horizon // 5
    indexes = list(range(1, prediction_index + 1))

    for col in what_if_cols:
        for i in indexes:
            new_col_name = col + "_what_if_" + str(i * 5)
            what_if_df = pd.concat([what_if_df, df[col].shift(-i).rename(new_col_name)], axis=1)
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

    processed_df = pd.DataFrame()

    for subject_id in subject_ids:
        # Identify the subset of the DataFrame for the current subject ID
        subset_df = df[df['id'] == subject_id]

        # Add time-lagged features directly to the subset
        lagged_features = add_time_lagged_features(subset_df, model_config_manager.get_num_features(),
                                                   model_config_manager.get_num_lagged_features())
        # Add what-if features
        what_if_df = add_what_if_features(subset_df, model_config_manager.get_what_if_features(),
                                          model_config_manager.get_prediction_horizon())

        # Update the subset DataFrame with the new time-lagged features
        subset_df = pd.concat([subset_df, lagged_features], axis=1)
        subset_df = pd.concat([subset_df, what_if_df], axis=1)

        # Adding new rows to the processed_df for this subset of the df
        processed_df = pd.concat([processed_df, subset_df], axis=0)

    if real_time:
        processed_df = processed_df.dropna(subset=processed_df.columns.difference(['target']))
    else:
        processed_df = processed_df.dropna()

    return processed_df
