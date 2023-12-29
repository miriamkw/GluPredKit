import pandas as pd
from glupredkit.models.loop import Model
import time
from glupredkit.metrics.rmse import Metric

file_path = "data/raw/" + "tidepool_11-12-2023_to_25-12-2023.csv"
df = pd.read_csv(file_path, index_col="date", parse_dates=True)
df = df.tail(1000)

loop_model = Model(prediction_horizon=30)

#loop_model.plot_prediction(df)

start_time = time.time()

predictions = loop_model.predict(df)

# Record the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

print(f"The function took {duration} seconds to complete.")

# print(predictions)
