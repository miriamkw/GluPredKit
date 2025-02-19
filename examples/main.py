import glupredkit.api as gpk
import numpy as np
from glupredkit.preprocessors.basic import Preprocessor
from glupredkit.models import ZeroOrderHold, NaiveLinearRegressor, Ridge
from glupredkit.plots import (AllMetricsTable, CGPMTable, ConfusionMatrix, ErrorGridPlot, ErrorGridTable, ParetoFrontier,
                              ResultsAcrossRegions, ScatterPlot, SinglePredictionHorizon, Trajectories,
                              TrajectoriesWithEvents, WeightedLoss)
from glupredkit.helpers.unit_config_manager import unit_config_manager

# Model configurations
prediction_horizon = 60
lookback = 12  # Lookback is given in number of time steps
numerical_features = ['CGM', 'insulin', 'carbs']
categorical_features = ['hour']
what_if_features = []
subject_ids = []  # Add subject ids if you want to use only a subset of the subjects in the dataset
show_plots = False  # Whether to show plots in a window. If false, they will still be saved to data/figures/
wandb_project = None  # If you want to log figures in weights and biases, change to project name

# Set whether to use mg/dL in visualizations and results
unit_config_manager.use_mgdl = True

# Get data in the standardized, parsed format
data = gpk.get_synthetic_data()
# data = gpk.get_parsed_data('OhioT1DM.csv')  # Assumes the data is located in data/raw/

# Process data
train_data, test_data = Preprocessor(subject_ids=subject_ids,
                                     numerical_features=numerical_features,
                                     categorical_features=categorical_features,
                                     what_if_features=what_if_features,
                                     prediction_horizon=prediction_horizon,
                                     num_lagged_features=lookback).__call__(data, add_time_lagged_features=True,
                                                                            add_what_if_features=True, dropna=True)
x_train, y_train = gpk.features_target_split(train_data)
zero_order_hold = ZeroOrderHold(prediction_horizon=prediction_horizon)
naive_linear_regressor = NaiveLinearRegressor(prediction_horizon=prediction_horizon)
ridge = Ridge(prediction_horizon=prediction_horizon).fit(x_train, y_train)

# Get results
x_test, y_test = gpk.features_target_split(test_data)
y_pred = zero_order_hold.predict(x_test)
zero_order_results = gpk.get_results_df('Zero Order Hold', train_data, test_data, y_pred, prediction_horizon,
                                num_lagged_features=lookback, num_features=numerical_features,
                                cat_features=categorical_features, what_if_features=what_if_features)
y_pred = naive_linear_regressor.predict(x_test)
naive_linreg_results = gpk.get_results_df('Naive Linear Regressor', train_data, test_data, y_pred, prediction_horizon,
                                num_lagged_features=lookback, num_features=numerical_features,
                                cat_features=categorical_features, what_if_features=what_if_features)
y_pred = ridge.predict(x_test)
ridge_results = gpk.get_results_df('Ridge', train_data, test_data, y_pred, prediction_horizon,
                                num_lagged_features=lookback, num_features=numerical_features,
                                cat_features=categorical_features, what_if_features=what_if_features)

result_dfs = [zero_order_results, naive_linreg_results, ridge_results]
results_figures = []
for ph in [30, 60]:
    results_figures += [
        AllMetricsTable().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        CGPMTable().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        ConfusionMatrix().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        ErrorGridPlot().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph, type='parkes'),
        ErrorGridTable().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        ParetoFrontier().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        ResultsAcrossRegions().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        ScatterPlot().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        SinglePredictionHorizon().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
        WeightedLoss().__call__(dfs=result_dfs, show_plot=show_plots, prediction_horizon=ph),
    ]
results_figures += [
    Trajectories().__call__(dfs=result_dfs, show_plot=show_plots),
    TrajectoriesWithEvents().__call__(dfs=result_dfs, show_plot=show_plots),
]

# Flatten the lists
plots, names = zip(*results_figures)
plots = [plot for sublist in plots for plot in sublist]
names = [name for sublist in names for name in sublist]

# Save figures in data/figures/
gpk.save_figures(plots, names)

if wandb_project:
    gpk.log_figures_in_wandb(wandb_project, plots, names)

