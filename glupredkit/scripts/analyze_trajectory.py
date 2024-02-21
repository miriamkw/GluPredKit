import matplotlib.pyplot as plt
import numpy as np
import re  # For regular expression operations
from glupredkit.metrics.rmse import Metric
from glupredkit.helpers.unit_config_manager import unit_config_manager
from glupredkit.helpers import cli as helpers
from glupredkit.helpers.model_config_manager import ModelConfigurationManager


def draw_slope_across_trajectory(model_file):
    # Fetching y values from a placeholder calculate_metrics function
    y_values = calculate_slope(model_file)
    x_values = list(range(5, 5 * len(y_values) + 1, 5))

    # Plotting
    plt.plot(x_values, y_values, marker='o')  # Example: Using 'o' as marker for each point

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'Slopes for {model_file}')
    plt.xlabel('Prediction Horizons')
    plt.ylabel(f'Slope in scatter plot')  # Placeholder for unit

    # Display the plot
    plt.show()


def draw_metric_across_trajectory(model_file, metric_file):
    # Fetching y values from a placeholder calculate_metrics function
    y_values = calculate_metrics(model_file, metric_file)
    x_values = list(range(5, 5 * len(y_values) + 1, 5))

    # Plotting
    plt.plot(x_values, y_values, marker='o')  # Example: Using 'o' as marker for each point

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'{metric_file} for {model_file}')
    plt.xlabel('Prediction Horizons')
    plt.ylabel(f'{metric_file} in {unit_config_manager.get_unit_string()}')  # Placeholder for unit

    # Display the plot
    plt.show()


def transform_feature_name(feature_names, prefix="insulin"):
    transformed_values = []
    for name in feature_names:
        if name == prefix:
            value = 0
        else:
            match = re.search(r'(\d+)$', name)
            if match:
                num = int(match.group())
                if "what_if" in name:
                    value = num
                else:
                    value = -num
            else:
                value = None  # or some default value if no match is found
        transformed_values.append(value)
    return transformed_values


def draw_coefficients_across_outputs(model_file, column_prefix="insulin"):
    # Load the trained model
    model_instance = helpers.get_trained_model(model_file)
    beta = model_instance.beta
    feature_names = model_instance.feature_names  # Assuming this exists and matches beta's rows

    # Identify indices of columns starting with the specified prefix (e.g., "insulin")
    relevant_indices = [i for i, name in enumerate(feature_names) if name.startswith(column_prefix)]
    relevant_feature_names = [name for name in feature_names if name.startswith(column_prefix)]

    feature_numbers = transform_feature_name(relevant_feature_names, column_prefix)
    feature_numbers = np.array(feature_numbers, dtype=float)

    # Extract the relevant coefficients for plotting
    relevant_coefficients = beta[relevant_indices, :]

    print(feature_numbers)

    # Sort feature_numbers and relevant_coefficients together
    sorted_indices = np.argsort(feature_numbers)
    sorted_feature_numbers = feature_numbers[sorted_indices]
    sorted_relevant_coefficients = relevant_coefficients[sorted_indices, :]

    # Plotting
    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
    for output_index in range(0, relevant_coefficients.shape[1], 4):
        # Plot coefficients for each output
        plt.plot(sorted_feature_numbers, sorted_relevant_coefficients[:, output_index], marker='o',
                 label=f'Output {output_index + 1}')

    plt.title(f"Coefficients for features starting with '{column_prefix}' across outputs")
    plt.xlabel('Time Offset')
    plt.ylabel('Coefficient Value')
    plt.xticks(rotation=45)  # Rotate feature names for readability
    plt.legend()
    plt.tight_layout()  # Adjust layout to make room for the legend and rotated x-axis labels
    plt.show()


def calculate_metrics(model_file, metric_file):
    """
    This command stores a report of the given metrics in data/reports/.
    """
    results = []
    model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                        int(model_file.split('__')[2].split('.')[0]))

    model_config_manager = ModelConfigurationManager(config_file_name)
    model_instance = helpers.get_trained_model(model_file)
    _, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)

    processed_data = model_instance.process_data(test_data, model_config_manager, real_time=False)

    target_columns = [column for column in processed_data.columns if column.startswith('target')]
    x_test = processed_data.drop(target_columns, axis=1)

    for index, target_column in enumerate(target_columns):
        # print("TARGET: ", target_column)
        # TODO: Add a test that targets are increasing, although I printed to verify
        y_pred = model_instance.predict(x_test)
        result = metric_file(processed_data[target_column], y_pred[:, index])
        results.append(result)

    return results


def calculate_slope(model_file):
    """
    This command stores a report of the given metrics in data/reports/.
    """
    results = []
    model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                        int(model_file.split('__')[2].split('.')[0]))

    model_config_manager = ModelConfigurationManager(config_file_name)
    model_instance = helpers.get_trained_model(model_file)
    _, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)

    processed_data = model_instance.process_data(test_data, model_config_manager, real_time=False)

    target_columns = [column for column in processed_data.columns if column.startswith('target')]
    x_test = processed_data.drop(target_columns, axis=1)

    for index, target_column in enumerate(target_columns):
        # print("TARGET: ", target_column)
        # TODO: Add a test that targets are increasing, although I printed to verify
        y_pred = model_instance.predict(x_test)
        slope, intercept = np.polyfit(processed_data[target_column], y_pred[:, index], 1)
        results.append(slope)

    return results


# Use the plots
metric = Metric()
model = 'ridge_multioutput_constrained__me_multioutput__180.pkl'

draw_slope_across_trajectory(model)
draw_metric_across_trajectory(model, metric)
# draw_coefficients_across_outputs(model, 'insulin')
