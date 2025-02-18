import glupredkit.api as gpk
from glupredkit.preprocessors.llm import Preprocessor
from glupredkit.models.zero_order import Model as ZeroOrder
from glupredkit.plots import ErrorGridPlot, AllMetricsTable, ConfusionMatrix
from glupredkit.helpers.unit_config_manager import unit_config_manager

# Model configurations
prediction_horizon = 60
lookback = 12  # Lookback is given in number of time steps
numerical_features = ['CGM', 'insulin', 'carbs']
categorical_features = ['hour']
what_if_features = []
subject_ids = []  # Add subject ids if you want to use only a subset of the subjects in the dataset
show_plots = True  # Whether to show plots in a window. If false, they will still be saved to data/figures/
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
                                     num_lagged_features=lookback).__call__(data)
x_train, y_train = gpk.features_target_split(train_data)
model = ZeroOrder(prediction_horizon=prediction_horizon)

# Test a model
x_test, y_test = gpk.features_target_split(test_data)
y_pred = model.predict(x_test)

# Get results dataframe to use in visualizations
results_df = gpk.get_results_df('Zero Order Hold', train_data, test_data, y_pred, prediction_horizon,
                                num_lagged_features=lookback, num_features=numerical_features,
                                cat_features=categorical_features, what_if_features=what_if_features)

# Draw plots
error_grid_plot = ErrorGridPlot()
metrics_table = AllMetricsTable()
confusion_matrix = ConfusionMatrix()

results_figures = []
for ph in [30, 60]:
    results_figures += [
        error_grid_plot(dfs=[results_df], show_plot=show_plots, prediction_horizon=ph, type='parkes'),
        metrics_table(dfs=[results_df], show_plot=show_plots, prediction_horizon=ph),
        confusion_matrix(dfs=[results_df], show_plot=show_plots, prediction_horizon=ph)
    ]

# Flatten the lists
plots, names = zip(*results_figures)
plots = [plot for sublist in plots for plot in sublist]
names = [name for sublist in names for name in sublist]

# Save figures in data/figures/
gpk.save_figures(plots, names)

if wandb_project:
    gpk.log_figures_in_wandb(wandb_project, plots, names)

