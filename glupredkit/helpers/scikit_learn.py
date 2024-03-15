import pandas as pd
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
import numpy as np
import matplotlib.pyplot as plt

def add_time_lagged_features(col_name, df, num_lagged_features):
    indexes = list(range(1, num_lagged_features))

    for i in indexes:
        new_col_name = col_name + "_" + str(i * 5)
        df = pd.concat([df, df[col_name].shift(i).rename(new_col_name)], axis=1)
    return df


def add_what_if(df, col_name, prediction_horizon):
    max_index = prediction_horizon // 5
    if prediction_horizon % 5 != 0:
        raise ValueError("Prediction horizon must be divisible by 5.")
    df = df.copy()
    for i in range(1, max_index + 1):
        new_col_name = f'{col_name}_what_if_{i * 5}'
        df = pd.concat([df, df[col_name].shift(-i).rename(new_col_name)], axis=1)
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

    # Add what-if features
    for col in model_config_manager.get_what_if_features():
        df = add_what_if(df, col, model_config_manager.get_prediction_horizons()[0])

    pred_indexes = model_config_manager.get_prediction_horizons()[0] // 5

    # Delete values where anywhere in predicted trajectory the cob is > 0
    for i in range(pred_indexes):
        df['cob'] = df['cob'].shift(-i)
        df = df[df['cob'] == 0]

    df['iob'].hist(bins=60, alpha=0.5)
    df['CGM'].hist(bins=60, alpha=0.5)
    plt.show()

    carbs_columns = [col for col in df.columns if col.startswith('carbs')]

    df = df.drop(columns=['iob', 'cob'] + carbs_columns)

    if real_time:
        df = df.dropna(subset=df.columns.difference(['target']))
    else:
        df = df.dropna()

    print(df.columns)

    print(df[['insulin_10', 'insulin_5', 'insulin', 'insulin_what_if_5', 'insulin_what_if_10', 'CGM', 'target_5', 'target_10']].tail(50))


    return df

