import pandas as pd
from src.models.gradient_boosting_regressor import GradientBoostingRegressor
from src.metrics.rmse import RMSE
from src.plots.prediction_trajectories import PredictionTrajectories
from src.plots.interactive_prediction import InteractivePrediction
import time
import pickle

df_train = pd.read_csv('data/train.csv', index_col=0)
df_test = pd.read_csv('data/test.csv', index_col=0)
df_current = pd.read_csv('data/current.csv', index_col=0)

start_time = time.time()

model = GradientBoostingRegressor()
model.fit(df_train)

end_time = time.time()
elapsed_time = end_time-start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")

#Save model
with open('gb.pkl', 'wb') as file:
    pickle.dump(model, file)





# Print error metrics
offset = 60
interval = 30
model_index = int(offset/interval) - 1
y_true = df_test['target_' + str(offset)]

#y_pred_constrained = model_constrained.predict(df_test, include_measurements=True)
y_pred = model.predict(df_test, include_measurements=True)

rmse = RMSE()

#print("RMSE constrained: ", rmse(y_true, [el[int(offset/interval)] for el in y_pred_constrained]))
print("RMSE: ", rmse(y_true, [el[int(offset/interval)] for el in y_pred]))







# Visualization
day = 1
use_mgdl = False

plot = PredictionTrajectories()
plot._draw_plot(y_pred[24*12*day:24*12*(day+1)], interval=interval, use_mgdl=use_mgdl)

plot = InteractivePrediction()
plot(model, df_current, interval=30, use_mgdl=use_mgdl, history_samples=int(12*3))





