from .base_plot import BasePlot
import ast
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glupredkit.helpers.unit_config_manager import unit_config_manager
from datetime import datetime

class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, start_index=12*12*1, n_samples=12*12, trajectory_interval=3, *args):
        # TODO: Fix the plot when using max samples in evaluate model
        """
        This plot plots predicted trajectories from the measured values. A random subsample of around 24 hours will
        be plotted.
        """
        # Choose the data from the first subject in dataset

        def on_hover(event):
            if event.inaxes == ax1:
                for hover_line in lines:
                    contains, _ = hover_line.contains(event)
                    if contains:
                        hover_line.set_alpha(1.0)
                    else:
                        hover_line.set_alpha(0.5)
                fig.canvas.draw_idle()

        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
        else:
            unit = "mmol/L"

        for df in dfs:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(14, 10))

            model_name = df['Model Name'][0]
            ph = int(df['prediction_horizon'][0])
            prediction_horizons = range(5, ph + 1, 5)

            dates_string_list = df['test_input_date'][0]
            datetime_strings = re.findall(r"'(.*?)'", dates_string_list)
            timestamp_list = [pd.Timestamp(ts) for ts in datetime_strings]

            # TODO: Only use the types that are present in the input
            # TODO: We only use the first 2000 elements to use data from only one subject...
            # Create a DataFrame
            try:
                model_df = pd.DataFrame({
                    'date': timestamp_list[:2000],
                    'CGM': get_list_from_string(df, 'test_input_CGM')[:2000],
                    'basal': get_list_from_string(df, 'test_input_basal')[:2000],
                    'bolus': get_list_from_string(df, 'test_input_bolus')[:2000],
                    'carbs': get_list_from_string(df, 'test_input_carbs')[:2000],
                    #'exercise': get_list_from_string(df, 'test_input_exercise')[:2000],
                })
            except Exception as e:
                print(f"Could not draw trajectories_with_events for {model_name}. Following feature in the model is "
                      f"required to draw this plot: {e}")
                continue  # Skip to the next iteration

            pred_cols = [col for col in df.columns if col.startswith('y_pred_')]
            for col in pred_cols:
                model_df[col] = get_list_from_string(df, col)[:2000]

            model_df.set_index('date', inplace=True)
            full_time_range = pd.date_range(start=model_df.index.min(), end=model_df.index.max(), freq='5T')

            # TODO: We need to fix these plots because they are not designed to handle different subjects in one df
            model_df = model_df.reindex(full_time_range)

            if n_samples > model_df.shape[0]:
                n_samples = model_df.shape[0]

            if start_index is not None:
                print("start index true")
                if start_index > model_df.shape[0] - n_samples:
                    print(f"Start index too high. Should be below {model_df.shape[0] - n_samples}. Setting start index to 0...")
                    start_index = 0
            elif model_df.shape[0] > n_samples:
                start_index = random.randint(0, model_df.shape[0] - n_samples)
            else:
                start_index = 0

            # Filter model df based on start index and num samples
            model_df = model_df.iloc[start_index:start_index + n_samples]

            y_pred_lists = []
            for prediction_horizon in prediction_horizons:
                y_pred = model_df[f'y_pred_{prediction_horizon}'].tolist()
                y_pred_lists += [y_pred]

            y_pred_lists = np.array(y_pred_lists)
            y_pred_lists = np.transpose(y_pred_lists)

            total_time = n_samples * 5
            t = np.arange(0, total_time, 5)

            # First plot (ax1)
            # Use correct unit
            # TODO: Make sure that both units works for this plot
            y_true = model_df['CGM'].tolist()
            if unit_config_manager.use_mgdl:
                ax1.axhspan(70, 180, facecolor='blue', alpha=0.2)
            else:
                y_true = [unit_config_manager.convert_value(val) for val in y_true]
                ax1.axhspan(unit_config_manager.convert_value(70), unit_config_manager.convert_value(180),
                           facecolor='blue', alpha=0.2)

            """
            plt.rcParams.update({
                'font.size': 20,  # General font size
                'xtick.labelsize': 20,  # Font size for x-axis tick labels
                'ytick.labelsize': 20,  # Font size for y-axis tick labels
                'axes.titlesize': 20,  # Font size for axes titles
                'axes.labelsize': 20  # Font size for axes labels
            })
            """

            ax1.set_title(f'Predicted trajectories for {model_name}', fontsize=18)
            ax1.set_ylabel(f'Blood glucose [{unit}]')

            ax1.scatter(t, y_true, label='Measurements', color='black')
            ax1.set_xlim(0, n_samples * 5)

            # Manually set font size for axis ticks
            #plt.xticks(fontsize=12)
            #plt.yticks(fontsize=12)

            prediction_horizons = range(0, ph + 1, 5)
            lines = []

            # Add predicted trajectories
            for i in range(len(y_pred_lists)):
                if (i + trajectory_interval - 1) % trajectory_interval == 0:
                    # Cut off the last predicted trajectories
                    end_index = n_samples - i

                    # Adding the true measurement to the beginning of the trajectory
                    trajectory = np.insert(y_pred_lists[i], 0, model_df['CGM'].tolist()[i])[:end_index]
                    x_vals = [val + i * 5 for val in prediction_horizons][:end_index]

                    if not unit_config_manager.use_mgdl:
                        trajectory = [unit_config_manager.convert_value(val) for val in trajectory]

                    # Only adding label to the first prediction to avoid duplicates
                    if i == 0:
                        line, = ax1.plot(x_vals, trajectory, linestyle='--', label='Predictions', color='#26B0F1')
                    else:
                        line, = ax1.plot(x_vals, trajectory, linestyle='--', color='#26B0F1')
                    lines.append(line)

            # Second plot
            #ax2.set_title('Carbohydrates and Exercise', fontsize=14)
            ax2.set_title('Carbohydrates', fontsize=18)

            for col in [col for col in df.columns if col.startswith('test_input')]:
                if 'basal' in col:
                    ax3.step(t, model_df['basal'].tolist(), label='Basal', where='mid')
                    ax3.set_ylabel('Basal Rate [U/hr]')
                    ax3.set_xlim(0, n_samples * 5)

                elif 'bolus' in col:
                    bolus = ax3.twinx()
                    bolus.bar(t, model_df['bolus'].tolist(), label='Bolus Doses', color='pink', width=5)
                    bolus.set_ylabel('Bolus Doses [IU]')

                elif 'exercise' in col:
                    ax2.bar(t, model_df['exercise'].tolist(), label='Exercise', width=5)
                    ax2.set_ylabel('Exercise [1-10]')
                    ax2.set_ylim(0, 10)  # Force the y-axis scale to be between 0 and 10
                    ax2.set_xlim(0, n_samples * 5)

                elif 'carbs' in col:
                    #carbs = ax2.twinx()
                    ax2.bar(t, model_df['carbs'].tolist(), label='Carbohydrates', color='green', width=5)
                    ax2.set_ylabel('Carbohydrates [grams]')

            # Third plot
            ax3.set_title('Insulin Inputs', fontsize=18)
            ax3.set_xlabel('Time (minutes)', fontsize=16)

            # Remove ticks on first and second plot
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])

            fig.canvas.mpl_connect('motion_notify_event', on_hover)

            # Collect all legends into one frame
            fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))  # Adjust position as needed
            file_path = "data/figures/"

            timestamp = datetime.now().isoformat()
            safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
            safe_timestamp = safe_timestamp.replace('.', '_')
            file_name = f"trajectories_{model_name}_{safe_timestamp}.png"
            plt.savefig(file_path + file_name)

            plt.show()


def get_list_from_string(df, col):
    string_values = df[col][0]
    string_values = string_values.replace("nan", "None")
    list_values = ast.literal_eval(string_values)
    list_values = [np.nan if x is None else x for x in list_values]
    return list_values

