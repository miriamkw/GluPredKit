import pandas as pd


def add_time_lagged_features(col_name, df, num_lagged_features):
    indexes = list(range(1, num_lagged_features + 1))

    for i in indexes:
        new_col_name = col_name + "_" + str(i * 5)
        df = pd.concat([df, df[col_name].shift(i).rename(new_col_name)], axis=1)
    return df


def process_data(df, num_lagged_features, numerical_features, categorical_features):

    # Add time-lagged features
    for col in numerical_features:
        df = add_time_lagged_features(col, df, num_lagged_features)

    df = df.dropna()

    return df

