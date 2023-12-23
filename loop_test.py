import pandas as pd
from glupredkit.models.loop_old import Model

file_path = "data/raw/" + "tidepool_09-12-2023_to_23-12-2023.csv"
df = pd.read_csv(file_path, index_col="date", parse_dates=True)

loop_model = Model(prediction_horizon=30)

#loop_model.one_prediction_testing(df)

loop_model.plot_prediction_contributions(df)

