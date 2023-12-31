import pandas as pd
import numpy as np
from glupredkit.helpers.model_config_manager import ModelConfigurationManager


def prepare_sequences(data, labels, window_size, real_time, step_size=12):
    X, y, dates = [], [], []
    exclude_list = ["target", "imputed"]
    sequence_columns = [item for item in data.columns if item not in exclude_list]

    # TODO: Implement what-if features
    for i in range(0, len(data) - window_size, step_size):
        sequence = data[sequence_columns][i:i + window_size]
        label = labels.iloc[i + window_size - 1]
        date = labels.index[i + window_size - 1]

        if 'imputed' in data.columns:
            imputed = data['imputed'].iloc[i + window_size - 1]
            if imputed:
                continue  # Skip this sequence

        if not real_time:
            if pd.isna(label):
                continue  # Skip this sequence

        X.append(sequence)
        y.append(label)
        dates.append(date)

    return np.array(X), np.array(y), dates


def create_dataframe(sequences, targets, dates):
    # Convert sequences to lists
    sequences_as_strings = [str(list(map(list, seq))) for seq in sequences]

    dataset_df = pd.DataFrame({
        'date': dates,
        'sequence': sequences_as_strings,  # sequences_as_lists,
        'target': targets
    })
    return dataset_df.set_index('date')


def process_data(df, model_config_manager: ModelConfigurationManager, real_time=False):

    df_X, df_y = df.drop("target", axis=1), df["target"]

    # Add sliding windows of features
    sequences, targets, dates = prepare_sequences(df_X, df_y,
                                                  window_size=model_config_manager.get_num_lagged_features(),
                                                  real_time=real_time)

    # Store as a dataframe with two columns: targets and sequences
    df = create_dataframe(sequences, targets, dates)

    return df

