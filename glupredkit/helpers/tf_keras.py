import pandas as pd
import numpy as np
from glupredkit.helpers.model_config_manager import ModelConfigurationManager


def prepare_sequences(data, labels, window_size, step_size=1):
    X, y, dates = [], [], np.array(labels.index)[0:len(data) - window_size]

    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:i + window_size])
        y.append(labels.iloc[i + window_size])

    return np.array(X), np.array(y), dates


def create_dataframe(sequences, targets, dates):
    # Create a new DataFrame to store sequences and targets

    # Convert sequences to lists
    sequences_as_strings = [str(list(map(list, seq))) for seq in sequences]

    dataset_df = pd.DataFrame({
        'date': dates,
        'sequence': sequences_as_strings,  # sequences_as_lists,
        'target': targets
    })
    dataset_df.set_index('date')
    return dataset_df


def process_data(df, model_config_manager: ModelConfigurationManager):

    df = df.dropna()

    df_X, df_y = df.drop("target", axis=1), df["target"]

    # Add sliding windows of features
    sequences, targets, dates = prepare_sequences(df_X, df_y,
                                                  window_size=model_config_manager.get_num_lagged_features())

    # Store as a dataframe with two columns: targets and sequences
    df = create_dataframe(sequences, targets, dates)

    return df

