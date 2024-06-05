from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('gRMSE')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        pen = [penalty(true_val, pred_val) for true_val, pred_val in zip(y_true, y_pred)]
        se = np.square(y_true - y_pred)
        gMSE = np.nanmean(se * pen)
        gRMSE = np.sqrt(gMSE)

        if unit_config_manager.use_mgdl:
            return gRMSE
        else:
            return unit_config_manager.convert_value(gRMSE)


def sigmoid(x, a, epsilon):
    # The xi function is defined wrongly in the paper, with 2 / epsilon in the end
    xi = (2 / epsilon) * (x - a - (epsilon / 2))
    if x <= a:
        return 0
    elif a < x <= a + (epsilon / 2):
        return -0.5 * xi ** 4 - xi ** 3 + xi + 0.5
    elif a + (epsilon / 2) < x <= a + epsilon:
        return 0.5 * xi ** 4 - xi ** 3 + xi + 0.5
    else:  # x > a + epsilon
        return 1


def sigmoid_hat(x, a, epsilon):
    # The xi function is defined wrongly in the paper, with 2 / epsilon in the end
    xi_hat = - (2 / epsilon) * (x - a + (epsilon / 2))
    if x <= a - epsilon:
        return 1
    elif a - epsilon < x <= a - (epsilon / 2):
        return 0.5 * xi_hat ** 4 - xi_hat ** 3 + xi_hat + 0.5
    elif a - (epsilon / 2) < x <= a:
        return -0.5 * xi_hat ** 4 - xi_hat ** 3 + xi_hat + 0.5
    else:  # a <= x
        return 0


def penalty(g, g_hat):
    # Constants from the table
    alpha_L = 1.5
    alpha_H = 1
    beta_L = 30
    beta_H = 100
    gamma_L = 10
    gamma_H = 20
    T_L = 85
    T_H = 155

    # Calculate the terms using the sigma functions
    sigma_T_L = sigmoid_hat(g, T_L, beta_L)
    sigma_gamma_L = sigmoid(g_hat, g, gamma_L)
    sigma_T_H = sigmoid(g, T_H, beta_H)
    sigma_gamma_H = sigmoid_hat(g_hat, g, gamma_H)

    # Final penalty calculation
    pen = (1 + alpha_L * sigma_T_L * sigma_gamma_L + alpha_H * sigma_T_H * sigma_gamma_H)
    return pen


"""
def plot_penalty():
    # Create a grid for g and g_hat
    g_values = np.linspace(0, 400, 400)
    g_hat_values = np.linspace(0, 400, 400)
    g_grid, g_hat_grid = np.meshgrid(g_values, g_hat_values)

    # Compute the penalty over the grid
    penalty_values = np.vectorize(penalty)(g_grid, g_hat_grid)

    # Define a custom color map
    colors = [(0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]  # Green -> Yellow -> Orange -> Red
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'custom_map'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Plot the contour plot of the penalty function
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(g_values, g_hat_values, penalty_values, cmap=custom_cmap, levels=np.linspace(1, 2.5, 100))
    plt.axline((0, 0), slope=1, color="white")
    plt.colorbar(contour)
    plt.title('Penalty Function Pen in the (g, ĝ) space')
    plt.xlabel('True Glucose Concentration, g [mg/dl]')
    plt.ylabel('Estimated Concentration, ĝ [mg/dl]')
    plt.show()


def plot_sigmoid(a, epsilon):
    # Create a grid for g and g_hat
    x_values = np.linspace(0, 400, 400)
    sigmoid_values = [sigmoid(x, a, epsilon) for x in x_values]

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, sigmoid_values, label='Sigmoid Function')
    plt.title('Sigmoid Function')
    plt.xlabel('x values')
    plt.ylabel('Sigmoid(x)')
    plt.legend()
    plt.show()


def plot_sigmoid_hat(a, epsilon):
    # Create a grid for g and g_hat
    x_values = np.linspace(0, 400, 400)
    sigmoid_values = [sigmoid_hat(x, a, epsilon) for x in x_values]

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, sigmoid_values, label='Sigmoid Function')
    plt.title('Sigmoid Function')
    plt.xlabel('x values')
    plt.ylabel('Sigmoid(x)')
    plt.legend()
    plt.show()
"""
