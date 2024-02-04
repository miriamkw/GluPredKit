import os
import pandas as pd
from datetime import datetime
from glupredkit.helpers.unit_config_manager import unit_config_manager
from glupredkit.helpers.model_config_manager import ModelConfigurationManager, generate_model_configuration
import glupredkit.helpers.cli as helpers

# List of model names
model_names = ["arx", "double_lstm", "elastic_net", "huber", "lasso", "lstm", "lstm_pytorch", "my_GBT", "my_lstm", "my_mlp", "plsr_with_diff_data_process", "random_forest", "ridge", "stacked_mlp_and_plsr", "stacked_with_plsr", "svr_linear", "svr_rbf", "tcn_pytorch", "tcn"]

# List of pH values
# ph_values = [30, 60]

trained_models_path = "data/trained_models/"


# Prepare a list of metrics
metrics = ["rmse_period"]
results = []

for model_name in model_names:
    config_file_name, prediction_horizon = 'my_config', 60
    model_file = f'{model_name}__{config_file_name}__{prediction_horizon}.pkl'

    model_config_manager = ModelConfigurationManager(config_file_name)
    model_instance = helpers.get_trained_model(model_file)
    _, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)
    #print("test_data: %s" % test_data)
    processed_data = model_instance.process_data(test_data, model_config_manager, real_time=False)
    #print("processed_data: %s" % processed_data)
    x_test = processed_data.drop('target', axis=1)
    #print("x_test: %s" % x_test)
    y_test = processed_data['target']

    y_pred = model_instance.predict(x_test)

    for metric in metrics:
        print("this is the metric: ", metric)
        metric_module = helpers.get_metric_module(metric)
        chosen_metric = metric_module.Metric()
        score = chosen_metric(y_test, y_pred)
        results.append({'Model name': model_name,
                        'Configuration': config_file_name,
                        'Prediction horizon': prediction_horizon,
                        'Metric': metric,
                        'Score': score})

# Convert results to DataFrame and save as CSV
df_results = pd.DataFrame(results)
metric_results_path = 'data/reports/period/'
os.makedirs(metric_results_path, exist_ok=True)
results_file_name = 'rmse_at_specific_period.csv'

df_results.to_csv(metric_results_path + results_file_name, index=False)