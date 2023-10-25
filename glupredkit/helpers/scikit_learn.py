import pandas as pd
from glupredkit.helpers.model_config_manager import ModelConfigurationManager


def add_time_lagged_features(col_name, df, num_lagged_features):
    indexes = list(range(1, num_lagged_features + 1))

    for i in indexes:
        new_col_name = col_name + "_" + str(i * 5)
        df = pd.concat([df, df[col_name].shift(i).rename(new_col_name)], axis=1)
    return df


def process_data(df, model_config_manager: ModelConfigurationManager, real_time=False):

    # Add time-lagged features
    for col in model_config_manager.get_num_features():
        df = add_time_lagged_features(col, df, model_config_manager.get_num_lagged_features())

    if real_time:
        df = df.dropna(subset=df.columns.difference(['target']))
    else:
        df = df.dropna()

    return df

