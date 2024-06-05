import pandas as pd
import numpy as np
from glupredkit.helpers.model_config_manager import ModelConfigurationManager


def prepare_sequences(df_X, df_y, window_size, what_if_columns, prediction_horizon, real_time, step_size=1):
    X, y, dates = [], [], []
    target_columns = df_y.columns
    exclude_list = list(target_columns) + ["imputed", "iob", "cob", "carbs"]
    sequence_columns = [item for item in df_X.columns if item not in exclude_list]
    n_what_if = prediction_horizon // 5

    print("Preparing sequences...")

    for i in range(0, len(df_X) - window_size - n_what_if, step_size):
        label = df_y.iloc[i + window_size - 1]

        if df_X.iloc[i:i + window_size + n_what_if].isnull().any().any():
            continue  # Skip this sequence if there are NaN values in the input data

        if 'imputed' in df_X.columns:
            imputed = df_X['imputed'].iloc[i + window_size - 1]
            if imputed:
                continue  # Skip this sequence

        if not real_time:
            if pd.isna(label).any():
                continue  # Skip this sequence

        # sequence = df_X[sequence_columns][i:i + window_size]
        date = df_y.index[i + window_size - 1]

        # Initialize an empty DataFrame for mixed_sequence with the same columns as sequence
        mixed_sequence = pd.DataFrame(index=df_X[i:i + window_size + n_what_if].index, columns=sequence_columns).iloc[0:window_size + n_what_if]

        for col in sequence_columns:
            if col in what_if_columns:
                # For "what if" columns, extend the sequence
                mixed_sequence[col] = df_X[col][i:i + window_size + n_what_if]
            else:
                # Slice the original window size for other columns
                original_sequence = df_X[col][i:i + window_size]
                # Create a padding series with -1 values and the appropriate date index
                padding_index = df_X.index[i + window_size:i + window_size + n_what_if]
                padding_series = pd.Series([-1] * n_what_if, index=padding_index)
                # Concatenate the original sequence with the padding series
                full_sequence = pd.concat([original_sequence, padding_series])
                mixed_sequence[col] = full_sequence

        X.append(mixed_sequence)
        y.append(label.values.tolist())
        dates.append(date)

    return np.array(X), np.array(y), dates


def create_dataframe(sequences, targets, dates):
    # Convert sequences to lists
    sequences_as_strings = [str(list(map(list, seq))) for seq in sequences]
    targets_as_strings = [','.join(map(str, target)) for target in targets]

    dataset_df = pd.DataFrame({
        'date': dates,
        'sequence': sequences_as_strings,  # sequences_as_lists
        'target': targets_as_strings
    })
    return dataset_df.set_index('date')


def process_data(df, model_config_manager: ModelConfigurationManager, real_time=False):
    target_columns = [col for col in df.columns if col.startswith('target')]
    df_X, df_y = df.drop(target_columns, axis=1), df[target_columns]

    # Add sliding windows of features
    sequences, targets, dates = prepare_sequences(df_X, df_y, window_size=model_config_manager.get_num_lagged_features(),
                                                  what_if_columns=model_config_manager.get_what_if_features(),
                                                  prediction_horizon=model_config_manager.get_prediction_horizon(),
                                                  real_time=real_time)

    # Store as a dataframe with two columns: targets and sequences
    df = create_dataframe(sequences, targets, dates)

    return df

