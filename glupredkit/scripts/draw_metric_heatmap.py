import numpy as np
import matplotlib.pyplot as plt
from glupredkit.metrics.grmse import Metric as gRMSE
from glupredkit.metrics.combined_metrics import Metric as Metric2
import matplotlib.colors as mcolors
from glupredkit.helpers.unit_config_manager import unit_config_manager


def draw_heatmap(metric, metric2=None, min_val=0, max_val=600, steps=1):
    tick = 100
    if not unit_config_manager.use_mgdl:
        min_val = unit_config_manager.convert_value(min_val)
        max_val = unit_config_manager.convert_value(max_val)
        tick = 10
        steps = 0.1

    true_values = np.arange(min_val, max_val + steps, steps)
    pred_values = np.arange(min_val, max_val + steps, steps)

    matrix1 = np.zeros((len(pred_values), len(true_values)))
    matrix2 = np.zeros_like(matrix1)

    for i, true in enumerate(pred_values):
        for j, pred in enumerate(true_values):
            matrix1[i, j] = metric([pred], [true])
            if metric2:
                matrix2[i, j] = metric2([pred], [true])
            else:
                matrix2[i, j] = matrix1[i, j]

    # Normalize both matrices to the range [0, 1] if not already normalized
    matrix1_normalized = (matrix1 - np.min(matrix1)) / (np.max(matrix1) - np.min(matrix1))
    matrix2_normalized = (matrix2 - np.min(matrix2)) / (np.max(matrix2) - np.min(matrix2))

    # Calculate the average of the two metrics
    avg_matrix = (matrix1_normalized * 0.9 + matrix2_normalized * 0.1) / 2

    fig, ax = plt.subplots(figsize=(10, 8))

    # Define a custom colormap from green to yellow to red
    cdict = {
        'red': [(0.0, 0.0, 0.0),  # Green
                (0.25, 1.0, 1.0),  # Yellow
                (0.5, 1.0, 1.0),  # Orange
                (0.75, 1.0, 1.0),  # Red
                (1.0, 0.2, 0.2)],  # Dark Red

        'green': [(0.0, 0.8, 0.8),  # Green
                  (0.25, 1.0, 1.0),  # Yellow
                  (0.5, 0.5, 0.5),  # Orange
                  (0.75, 0.0, 0.0),  # Red
                  (1.0, 0.0, 0.0)],  # Dark Red

        'blue': [(0.0, 0.0, 0.0),  # Green
                 (0.25, 0.0, 0.0),  # Yellow
                 (0.5, 0.0, 0.0),  # Orange
                 (0.75, 0.0, 0.0),  # Red
                 (1.0, 0.0, 0.0)]  # Dark Red
    }

    custom_colormap = mcolors.LinearSegmentedColormap('CustomMap', cdict)
    c = ax.pcolormesh(true_values, pred_values, avg_matrix, shading='auto', cmap=custom_colormap)

    # Create a colorbar for the heatmap
    fig.colorbar(c, ax=ax)

    # Set labels and title
    ax.set_xlabel(f'True Values [{unit_config_manager.get_unit_string()}]')
    ax.set_ylabel(f'Predicted Values [{unit_config_manager.get_unit_string()}]')
    ax.set_title(f'{metric.name} Heatmap for True vs. Predicted Values')

    # Setting the aspect ratio to 'equal' makes the heatmap squares instead of rectangles
    ax.set_aspect('equal', 'box')

    # Adjust the limits to fit the data
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Generate ticks at intervals of 1 within the specified range
    ticks = np.arange(min_val, max_val, tick)

    # Adding ticks for clarity
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    plt.show()


# Example usage
# Initialize the metric
metric1 = Metric2()
metric2 = Metric2()
# Draw the heatmap
draw_heatmap(metric1)
