import pandas as pd
from src.models.ridge_regressor import RidgeRegressor
from src.metrics.rmse import RMSE
from src.metrics.mae import MAE
from src.plots.prediction_trajectories import PredictionTrajectories
import matplotlib.pyplot as plt
from src.plots.interactive_prediction import InteractivePrediction
from src.plots.compare_prediction_trajetories import ComparePredictionTrajectories
from src.plots.compare_scatter_plot import CompareScatterPlot
import time
import pickle

df_train = pd.read_csv('data/train.csv', index_col=0)
df_test = pd.read_csv('data/test.csv', index_col=0)
df_current = pd.read_csv('data/current.csv', index_col=0)


#print(df_train[['CGM', 'CGM_what_if_30', 'CGM_what_if_60', 'target_30', 'target_60']].head(40))

#model_constrained = RidgeRegressor(is_constrained=True)
#model_constrained.fit(df_train)

print("shape train: ", df_train.shape)
print("shape test: ", df_test.shape)

"""
start_time = time.time()

model = RidgeRegressor(is_constrained=True)
model.fit(df_train)

end_time = time.time()
elapsed_time = end_time-start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")
"""


# Save model
# with open('ridge_constrained.pkl', 'wb') as file:
#    pickle.dump(model, file)


#Load models
models = []
model_names = []

with open('ridge_basic.pkl', 'rb') as file:
#with open('ridge_basic.pkl', 'rb') as file:
    ridge_basic = pickle.load(file)

    for i in range(6):
        print("Ridge basic "+ str(i*30 + 30) + ":", ridge_basic.models[i].get_params()['ridge'].get_params()['alpha'])
    models.append(ridge_basic)
    model_names.append("OTS RR")

with open('ridge_constrained.pkl', 'rb') as file:
#with open('ridge_basic.pkl', 'rb') as file:
    ridge_con = pickle.load(file)
    for i in range(6):
        print("Ridge con " + str(i * 30 + 30) + ":", ridge_con.models[i].get_params()['ridge'].get_params()['alpha'])
    models.append(ridge_con)
    model_names.append("Proposed RR")

with open('gb.pkl', 'rb') as file:
#with open('ridge_basic.pkl', 'rb') as file:
    gb = pickle.load(file)

    for i in range(6):
        print("Gb min split " + str(i * 30 + 30) + ":", gb.models[i].get_params()['gradientboostingregressor'].get_params()['min_samples_split'])
        print("Gb leaf " + str(i * 30 + 30) + ":",
              gb.models[i].get_params()['gradientboostingregressor'].get_params()['min_samples_leaf'])

    models.append(gb)
    model_names.append("OTS GBR")

"""
# Print error metrics
offset = 60
interval = 30
model_index = int(offset/interval) - 1
y_true = df_test['target_' + str(offset)]

#y_pred_constrained = model_constrained.predict(df_test, include_measurements=True)
y_pred = model.predict(df_test, include_measurements=True)

rmse = RMSE(use_mgdl=False)

#print("RMSE constrained: ", rmse(y_true, [el[int(offset/interval)] for el in y_pred_constrained]))
print("RMSE: ", rmse(y_true, [el[int(offset/interval)] for el in y_pred]))


print("RMSE 30: ", rmse(y_true, [el[int(30/interval)] for el in y_pred]))
print("RMSE 60: ", rmse(y_true, [el[int(60/interval)] for el in y_pred]))
print("RMSE 90: ", rmse(y_true, [el[int(90/interval)] for el in y_pred]))
print("RMSE 120: ", rmse(y_true, [el[int(120/interval)] for el in y_pred]))
print("RMSE 150: ", rmse(y_true, [el[int(150/interval)] for el in y_pred]))
print("RMSE 180: ", rmse(y_true, [el[int(180/interval)] for el in y_pred]))
"""
rmse = MAE(use_mgdl=False)
interval = 30

for i in range(len(models)):
    print("MODEL NR: ", i)

    for offset in [30, 60, 90, 120, 150, 180]:
        model_index = int(offset / interval) - 1
        y_true = df_test['target_' + str(offset)]
        y_pred = models[i].predict(df_test, include_measurements=True)

        res = rmse._calculate_metric(y_true=y_true, y_pred=[el[int(offset / interval)] for el in y_pred])

        print("MAE " + str(offset) + ":", res)





# print(model.model_60.feature_names_in_)

# Print error metrics for with vs without constraints

# Get feature names from column transformer

model = ridge_con
model_index = 1 # The index of which we can access the model in the pipeline

font = {
    'size': 22,
}

plt.rc('font', **font)

feature_names = model.models[model_index][0].get_feature_names_out()
feature_names = [name.replace('num__', '') for name in feature_names]
feature_names = [name.replace('cat__', '') for name in feature_names]

sorted_indices = sorted(range(len(model.models[model_index][1].coef_)), key=lambda k: model.models[model_index][1].coef_[k], reverse=True)
coefs_sorted = [model.models[model_index][1].coef_[i] for i in sorted_indices]
feature_names_sorted = [feature_names[i] for i in sorted_indices]

for i in range(len(coefs_sorted)):
    print(feature_names_sorted[i] + ":", coefs_sorted[i])


model = gb
model_index = 1


# Access the feature importances
feature_importances = gb.models[model_index].get_params()['gradientboostingregressor'].feature_importances_

# Pair the feature importances with their corresponding feature names
feature_names = model.models[model_index][0].get_feature_names_out()
feature_names = [name.replace('num__', '') for name in feature_names]
feature_names = [name.replace('cat__', '') for name in feature_names]
feature_importances_dict = dict(zip(feature_names, feature_importances))

# Sort the features by importance in descending order
sorted_features = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)

# Print the top N most important features
top_n = 10  # Replace with the number of top features you want to display
print(f"Top {top_n} most important features:")
for feature, importance in sorted_features[:top_n]:
    print(f"{feature}: {importance}")


# Extract feature names and importances from sorted_features
feature_names = [feature[0] for feature in sorted_features[:top_n]]
importances = [feature[1] for feature in sorted_features[:top_n]]

# Plotting the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances)
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
plt.xlabel('Feature Name')
plt.ylabel('Feature Importance')
plt.title('OTS Gradient Boosting Regressor (GBR) Most Important Features')
plt.tight_layout()
plt.show()


"""
for i in range(len(coefs_sorted)):
    print(feature_names[i] + ":", model.models[model_index][1].coef_[i])

"""


# Plot the parameters like in notebook

#features = ['carbs', 'insulin', 'activity_state', 'CGM', 'hour']

"""
features = ['carbs', 'insulin']

fig, axs = plt.subplots(1, 2, figsize=(20, 5), sharey=True)

for index, feature in enumerate(features):
    ax = axs[index]

    x = []
    y = []

    for i in range(len(feature_names_sorted)):
        if feature_names_sorted[i].startswith(feature):
            if "avg_12h" in feature_names_sorted[i]:
                break
            elif "12h_hypo_count" in feature_names_sorted[i]:
                break
            elif "inverted" in feature_names_sorted[i]:
                break
            else:
                x.append(feature_names_sorted[i])
                y.append(coefs_sorted[i])

    for i in range(len(x)):
        if x[i] == feature:
            x[i] = 0
        elif "what_if" in x[i]:
            if feature == 'activity_state':
                x[i] = int(x[i].split("_")[4])
            else:
                x[i] = int(x[i].split("_")[3])
        else:
            if feature == 'activity_state':
                x[i] = int(x[i].split("_")[2])
            else:
                x[i] = -int(x[i].split("_")[1])

    # Sort x and rearrange y accordingly
    sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
    x_sorted = [x[i] for i in sorted_indices]
    y_sorted = [y[i] for i in sorted_indices]

    ax.scatter(x_sorted, y_sorted)
    ax.set(title='Feature: ' + feature, xlabel='Time lag (minutes)', ylabel='Coefficient')

# Remove y-axis ticks from the right subplot (index 1)
axs[1].set(ylabel='')

plt.suptitle("Proposed RR Coefficients")
plt.tight_layout()
plt.show()
"""


"""
features = ['carbs', 'insulin']

for feature in features:
    fig, ax = plt.subplots(figsize=(20, 5))

    x = []
    y = []

    for i in range(len(feature_names_sorted)):
        if feature_names_sorted[i].startswith(feature):
            if "avg_12h" in feature_names_sorted[i]:
                break
            elif "12h_hypo_count" in feature_names_sorted[i]:
                break
            elif "inverted" in feature_names_sorted[i]:
                break
            else:
                x.append(feature_names_sorted[i])
                y.append(coefs_sorted[i])

    for i in range(len(x)):
        if x[i] == feature:
            x[i] = 0
        elif "what_if" in x[i]:
            if feature == 'activity_state':
                x[i] = int(x[i].split("_")[4])
            else:
                x[i] = int(x[i].split("_")[3])
        else:
            if feature == 'activity_state':
                x[i] = int(x[i].split("_")[2])
            else:
                x[i] = -int(x[i].split("_")[1])

    # Sort x and rearrange y accordingly
    sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
    x_sorted = [x[i] for i in sorted_indices]
    y_sorted = [y[i] for i in sorted_indices]

    ax.scatter(x_sorted, y_sorted)
    ax.set(title='Feature: ' + feature)
    ax.set(xlabel='Time lag (minutes)', ylabel='Coefficient')

    plt.title("Proposed RR Coefficients")

    plt.show()

"""



# Visualization
day = 1
use_mgdl = False

y_pred = models[0].predict(df_current, include_measurements=True)

plot = PredictionTrajectories()
#plot._draw_plot(y_pred[24*12*day:24*12*(day+1)], interval=interval, use_mgdl=use_mgdl)
plot._draw_plot(y_pred[24:], interval=interval, use_mgdl=use_mgdl)


plot = InteractivePrediction()
#plot(model, df_current, interval=30, use_mgdl=use_mgdl, history_samples=int(12*3))

plot = ComparePredictionTrajectories()
#bolus 1227-1234
measurements = [3.719, 4.10755, 4.77364, 5.88379, 7.715539999999999, 8.49264, 9.43627, 9.6583, 9.21424, 8.82569, 8.71467, 8.60366, 8.43714, 8.38163, 8.32612, 8.1596, 8.10409, 8.10409, 8.10409, 8.10409, 8.1596, 7.93757, 7.993080000000001, 8.10409]




plot(models=models, model_names=model_names, measurements=measurements, df=df_current, interval=30, use_mgdl=use_mgdl, history_samples=int(12*1))


offset = 30

plot = CompareScatterPlot()

predictions = []
for mod in models:
    y_pred = mod.predict(df_test, include_measurements=True)
    predictions.append([el[int(offset / interval)] for el in y_pred])


y_true = df_test['target_' + str(offset)]
plot(predictions=predictions, model_names=model_names, y_true=y_true, use_mgdl=False, title=str(offset) + " minutes")

offsets = [60, 90, 120, 150, 180]

for offset in offsets:
    predictions = []
    for mod in models:
        y_pred = mod.predict(df_test, include_measurements=True)
        predictions.append([el[int(offset / interval)] for el in y_pred])

    plot(predictions=predictions, model_names=model_names, y_true=y_true, use_mgdl=False,
         title=str(offset) + " minutes")
         




