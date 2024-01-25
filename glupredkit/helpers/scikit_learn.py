import pandas as pd
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
import numpy as np

def add_time_lagged_features(col_name, df, num_lagged_features):
    indexes = list(range(1, num_lagged_features))

    for i in indexes:
        new_col_name = col_name + "_" + str(i * 5)
        df = pd.concat([df, df[col_name].shift(i).rename(new_col_name)], axis=1)
    return df


def process_data(df, model_config_manager: ModelConfigurationManager, real_time=False):

    if "imputed" in df.columns:
        # Temporarily convert 'imputed' to float
        df['imputed'] = df['imputed'].astype(float)
        # Set entire rows to NaN where 'imputed' is True
        df.loc[df['imputed'] == 1.0, :] = np.nan
        df = df.drop(columns=['imputed'])

    # Add time-lagged features
    for col in model_config_manager.get_num_features():
        df = add_time_lagged_features(col, df, model_config_manager.get_num_lagged_features())

    if real_time:
        df = df.dropna(subset=df.columns.difference(['target']))
    else:
        df = df.dropna()

    return df

