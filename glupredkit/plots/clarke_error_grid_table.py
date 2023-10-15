from .base_plot import BasePlot
from error_grids import zone_accuracy
import matplotlib.pyplot as plt
import os


class Plot(BasePlot):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

    def __call__(self, models_data, y_true):
        """
        Plots the scatter plot for the given trained_models data.

        models_data: A list of dictionaries containing the model name, y_true, and y_pred.
                    Example: [{'name': 'model1', 'y_pred': [...]}, ...]
        """
        headers = ['Model', 'Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']
        rows = []

        for model_data in models_data:
            model_name = model_data.get('name')
            y_pred = model_data.get('y_pred')

            accuracy_values = zone_accuracy(y_true, y_pred, 'clarke')
            formatted_values = ["{:.1f}%".format(value * 100) for value in accuracy_values]
            rows.append([model_name] + formatted_values)

        # Plotting the table
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust as per your requirements)
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')

        table.auto_set_font_size(False)  # Disable automatic font size adjustment
        table.scale(1, 4)  # You can adjust the scaling factors as needed

        # Bold header
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 0 is the row with headers
                cell.set_text_props(weight='bold', fontsize=16)
            else:
                cell.set_fontsize(14)  # Adjust as needed

        table.auto_set_column_width(col=0)

        # Save table as figure
        file_path = "data/figures/"
        os.makedirs(file_path, exist_ok=True)
        file_name = f'clarke_error_grid_table_ph-{self.prediction_horizon}.png'

        plt.title("Clarke Error Grid")

        plt.savefig(file_path + file_name)
        plt.show()
