import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import ast
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from glupredkit.helpers.unit_config_manager import unit_config_manager
from io import BytesIO
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
import seaborn as sns


def get_df_from_results_file(file_name):
    file_path = 'data/tested_models/' + file_name
    return pd.read_csv(file_path)


def generate_single_model_front_page(canvas, df):
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(100, 720, f'Model Configuration')

    canvas.setFont("Helvetica", 12)
    canvas.drawString(100, 700, f'Prediction horizon: {df["prediction_horizon"][0]} minutes')
    canvas.drawString(100, 680, f'Numerical features: {df["num_features"][0]}')
    canvas.drawString(100, 660, f'Categorical features: {df["cat_features"][0]}')
    canvas.drawString(100, 640, f'What-if features: {df["what_if_features"][0]}')
    canvas.drawString(100, 620, f'Number of time-lagged features: {df["num_lagged_features"][0]}')
    canvas.drawString(100, 600, f'Preprocessor: {df["preprocessor"][0]}')

    # Subtitle
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(100, 560, f'Data Description')

    if df["subject_ids"][0] == '[]':
        data_ids = "All"
    else:
        data_ids = df["subject_ids"][0]

    # Normal text
    canvas.setFont("Helvetica", 12)
    canvas.drawString(100, 540, f'Dataset: {df["data"][0]}')
    canvas.drawString(100, 520, f'Data ids: {data_ids}')
    canvas.drawString(100, 480, f'Training samples: {df["training_samples"][0]}')
    canvas.drawString(100, 460, f'Hypoglycemia training samples: {df["hypo_training_samples"][0]}')
    canvas.drawString(100, 440, f'Hyperglycemia training samples: {df["hyper_training_samples"][0]}')
    canvas.drawString(100, 400, f'Test samples: {df["test_samples"][0]}')
    canvas.drawString(100, 380, f'Hypoglycemia test samples: {df["hypo_test_samples"][0]}')
    canvas.drawString(100, 360, f'Hyperglycemia test samples: {df["hyper_test_samples"][0]}')

    return canvas


def set_title(canvas, title_string):
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawCentredString(letter[0] / 2, 750, title_string)
    return canvas


def set_subtitle(canvas, title_string, y_placement=720):
    # Subtitle
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(100, y_placement, title_string)
    return canvas


def set_bottom_text(canvas):
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(letter[0] / 2, 50, f'This report is generated using GluPredKit '
                                                f'(https://github.com/miriamkw/GluPredKit)')
    return canvas


def draw_model_accuracy_table(c, df):
    table_data = [
        ['Prediction Horizon', f'RMSE [{unit_config_manager.get_unit()}]',
         f'ME [{unit_config_manager.get_unit()}]', 'MARE [%]']
    ]
    table_interval = 30
    # TODO: Add some color coding in the table?
    prediction_horizon = int(df['prediction_horizon'][0])
    for ph in range(table_interval, prediction_horizon + 1, table_interval):
        rmse_str = "{:.1f}".format(float(df[f'rmse_{ph}'][0]))
        me_str = "{:.1f}".format(float(df[f'me_{ph}'][0]))
        mare_str = "{:.1f}".format(float(df[f'mare_{ph}'][0]))
        new_row = [[str(ph), rmse_str, me_str, mare_str]]
        table_data += new_row

    # Average values
    def get_avg(metric):
        metric_cols = [col for col in df.columns if col.startswith(metric)]
        avg_score = np.mean(df[metric_cols].iloc[0])
        return avg_score

    rmse_str = "{:.1f}".format(get_avg('rmse'))
    me_str = "{:.1f}".format(get_avg('me'))
    mare_str = "{:.1f}".format(get_avg('mare'))
    new_row = [['Average', rmse_str, me_str, mare_str]]
    table_data += new_row
    c = draw_table(c, table_data, 700 - 20 * int(df['prediction_horizon'][0]) // table_interval)
    return c


def draw_model_comparison_accuracy_table(c, dfs, metric, y_placement):
    data = [
        ['Rank', 'Model', f'Average {metric} [{unit_config_manager.get_unit()}]']
    ]

    models = []
    result_list = []
    for df in dfs:
        prediction_horizons = range(5, get_ph(df) + 1, 5)
        models += [df['Model Name'][0]]
        result_values = []
        for ph in prediction_horizons:
            result_values += [df[f'{metric}_{ph}'][0]]
        result_list += [np.mean(result_values)]

    # Sort for ranking
    pairs = zip(result_list, models)

    # Sort the pairs based on the RMSE values (in ascending order)
    sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=False)

    # Unpack the sorted pairs back into separate lists
    sorted_result_list, sorted_models_list = zip(*sorted_pairs)

    for i in range(len(models)):
        result_str = "{:.1f}".format(sorted_result_list[i])
        new_row = [f'#{i + 1}', sorted_models_list[i], result_str]
        data += [new_row]

    c = draw_table(c, data, y_placement)

    return c


def draw_model_comparison_error_grid_table(c, dfs, y_placement):
    data = [
        ['Rank', 'Model', f'Average Error Grid Score']
    ]

    models = []
    result_list = []
    for df in dfs:
        prediction_horizons = range(5, get_ph(df) + 1, 5)
        models += [df['Model Name'][0]]
        result_values = []
        for ph in prediction_horizons:
            result_values += [df[f'parkes_error_grid_exp_{ph}'][0]]
        result_list += [np.mean(result_values) * 100]

    # Sort for ranking
    pairs = zip(result_list, models)

    # Sort the pairs based on the error grid values (in descending order)
    sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Unpack the sorted pairs back into separate lists
    sorted_result_list, sorted_models_list = zip(*sorted_pairs)

    for i in range(len(models)):
        result_str = "{:.1f}%".format(sorted_result_list[i])
        new_row = [f'#{i + 1}', sorted_models_list[i], result_str]
        data += [new_row]

    c = draw_table(c, data, y_placement)

    return c


def draw_model_comparison_glycemia_detection_table(c, dfs, y_placement):
    data = [
        ['Rank', 'Model', f'MCC Hypo', 'MCC Hyper', 'Total MCC']
    ]

    models = []
    hypo_result_list = []
    hyper_result_list = []
    total_result_list = []
    for df in dfs:
        prediction_horizons = range(5, get_ph(df) + 1, 5)
        models += [df['Model Name'][0]]
        hypo_result_values = []
        hyper_result_values = []
        total_result_values = []
        for ph in prediction_horizons:
            hypo_result_values += [df[f'mcc_hypo_{ph}'][0]]
            hyper_result_values += [df[f'mcc_hyper_{ph}'][0]]
            total_result_values += [(df[f'mcc_hypo_{ph}'][0] + df[f'mcc_hyper_{ph}'][0]) / 2]
        hypo_result_list += [np.mean(hypo_result_values)]
        hyper_result_list += [np.mean(hyper_result_values)]
        total_result_list += [np.mean(total_result_values)]

    # Sort for ranking
    pairs = zip(hypo_result_list, hyper_result_list, total_result_list, models)

    # Sort the pairs based on the MCC values (in descending order)
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    # Unpack the sorted pairs back into separate lists
    sorted_hypo_list, sorted_hyper_list, sorted_total_list, sorted_models_list = zip(*sorted_pairs)

    for i in range(len(models)):
        hypo_result_str = "{:.1f}".format(sorted_hypo_list[i])
        hyper_result_str = "{:.1f}".format(sorted_hyper_list[i])
        total_result_str = "{:.1f}".format(sorted_total_list[i])
        new_row = [f'#{i + 1}', sorted_models_list[i], hypo_result_str, hyper_result_str, total_result_str]
        data += [new_row]

    c = draw_table(c, data, y_placement)

    return c


def draw_model_comparison_predicted_distribution_table(c, dfs, y_placement):
    data = [
        ['Rank', 'Model', 'Standard Deviation Predicted versus Measured']
    ]

    models = []
    result_list = []
    deviation_from_target_list = []
    for df in dfs:
        prediction_horizons = range(5, get_ph(df) + 1, 5)
        models += [df['Model Name'][0]]
        std_result_values = []
        for ph in prediction_horizons:
            y_test = df[f'target_{ph}'][0]
            y_pred = df[f'y_pred_{ph}'][0]
            y_test = ast.literal_eval(y_test)
            y_pred = y_pred.replace("nan", "None")
            y_pred = ast.literal_eval(y_pred)
            y_pred = [np.nan if val is None else val for val in y_pred]
            result = np.nanstd(y_pred) / np.nanstd(y_test) * 100
            std_result_values += [result]
        result_list += [np.mean(std_result_values)]
        deviation_from_target_list += [np.abs(np.mean(std_result_values) - 100)]

    # Sort for ranking
    pairs = zip(deviation_from_target_list, result_list, models)

    # Sort the pairs based on the MCC values (in descending order)
    sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=False)

    # Unpack the sorted pairs back into separate lists
    _, sorted_result_list, sorted_models_list = zip(*sorted_pairs)

    for i in range(len(models)):
        total_result_str = "{:.0f}%".format(sorted_result_list[i])
        new_row = [f'#{i + 1}', sorted_models_list[i], total_result_str]
        data += [new_row]

    c = draw_table(c, data, y_placement)

    return c


def draw_overall_ranking_table(c, dfs, y_placement):
    data = [
        ['Rank', 'Model', 'RMSE', 'ME', 'SEG', 'MCC', 'STD']
    ]
    # TODO: Sort ranking based on total rank entries
    models = []
    rmse_list = []
    me_list = []
    seg_list = []
    mcc_list = []
    std_list = []
    for df in dfs:
        prediction_horizons = range(5, get_ph(df) + 1, 5)
        models += [df['Model Name'][0]]
        rmse_list += [np.mean([df[f'rmse_{ph}'][0] for ph in prediction_horizons])]
        me_list += [np.mean([df[f'me_{ph}'][0] for ph in prediction_horizons])]
        seg_list += [np.mean([df[f'parkes_error_grid_exp_{ph}'][0] for ph in prediction_horizons])]
        mcc_list += [np.mean([(df[f'mcc_hypo_{ph}'][0] + df[f'mcc_hyper_{ph}'][0]) / 2 for ph in prediction_horizons])]

        y_test = [ast.literal_eval(df[f'target_{ph}'][0]) for ph in prediction_horizons]
        y_test_std = [np.nanstd(y_vals) for y_vals in y_test]

        y_pred_std = []
        for ph in prediction_horizons:
            y_pred = df[f'y_pred_{ph}'][0].replace("nan", "None")
            y_pred = ast.literal_eval(y_pred)
            y_pred = [np.nan if val is None else val for val in y_pred]
            result = np.nanstd(y_pred)
            y_pred_std += [result]

        relative_std = np.abs(np.mean(y_pred_std) / np.mean(y_test_std) * 100 - 100)
        std_list += [relative_std]

    results_df = pd.DataFrame({'Models': models,
                               'RMSE': rmse_list,
                               'ME': me_list,
                               'SEG': seg_list,
                               'MCC': mcc_list,
                               'STD': std_list,
                               })
    results_df['RMSE'] = results_df['RMSE'].rank(method='min')
    results_df['ME'] = results_df['ME'].abs().rank(method='min')
    results_df['SEG'] = results_df['SEG'].rank(method='min', ascending=False)
    results_df['MCC'] = results_df['MCC'].rank(method='min', ascending=False)
    results_df['STD'] = results_df['STD'].rank(method='min')
    results_df['total'] = (results_df['RMSE'] + results_df['ME'] + results_df['SEG'] + results_df['MCC']
                           + results_df['STD'])
    results_df = results_df.sort_values(by='total', ascending=True)

    models = results_df['Models'].tolist()
    rmse_list = results_df['RMSE'].tolist()
    me_list = results_df['ME'].tolist()
    seg_list = results_df['SEG'].tolist()
    mcc_list = results_df['MCC'].tolist()
    std_list = results_df['STD'].tolist()

    for i in range(len(models)):
        rmse_str = "#{:.0f}".format(rmse_list[i])
        me_str = "#{:.0f}".format(me_list[i])
        seg_str = "#{:.0f}".format(seg_list[i])
        mcc_str = "#{:.0f}".format(mcc_list[i])
        std_str = "#{:.0f}".format(std_list[i])
        new_row = [f'#{i + 1}', models[i], rmse_str, me_str, seg_str, mcc_str, std_str]
        data += [new_row]

    c = draw_table(c, data, y_placement)

    return c


def draw_error_grid_table(c, df):
    table_data = [
        ['Prediction Horizon', f'Zone A', f'Zone B', f'Zone C', f'Zone D', f'Zone E']
    ]
    table_interval = 30
    # TODO: Add some color coding in the table?
    for ph in range(table_interval, int(df['prediction_horizon'][0]) + 1, table_interval):
        new_row = [str(ph)]
        current_data = df[f'parkes_error_grid_{ph}'][0]
        current_data = ast.literal_eval(current_data)
        for i in range(5):
            new_row += [current_data[i]]
        table_data += [new_row]
    c = draw_table(c, table_data, 720 - 20 * int(df['prediction_horizon'][0]) // table_interval)
    return c


def draw_mcc_table(c, df):
    table_data = [
        ['Prediction Horizon', 'MCC Hypoglycemia', 'MCC Hyperglycemia', 'Average']
    ]
    table_interval = 30
    # TODO: Add some color coding in the table?
    prediction_horizon = int(df['prediction_horizon'][0])
    for ph in range(table_interval, prediction_horizon + 1, table_interval):
        mcc_hypo = float(df[f'mcc_hypo_{ph}'][0])
        mcc_hyper = float(df[f'mcc_hyper_{ph}'][0])
        mcc_hypo_str = "{:.2f}".format(mcc_hypo)
        mcc_hyper_str = "{:.2f}".format(mcc_hyper)
        mcc_avg_str = "{:.2f}".format(np.mean([mcc_hypo, mcc_hyper]))
        new_row = [[str(ph), mcc_hypo_str, mcc_hyper_str, mcc_avg_str]]
        table_data += new_row

    mcc_hypo_avg = np.mean([float(df[f'mcc_hypo_{ph}'][0]) for ph in range(5, prediction_horizon + 1, 5)])
    mcc_hyper_avg = np.mean([float(df[f'mcc_hyper_{ph}'][0]) for ph in range(5, prediction_horizon + 1, 5)])
    total_avg = np.mean([mcc_hypo_avg, mcc_hyper_avg])
    mcc_hypo_avg_str = "{:.2f}".format(mcc_hypo_avg)
    mcc_hyper_avg_str = "{:.2f}".format(mcc_hyper_avg)
    total_avg_str = "{:.2f}".format(total_avg)
    new_row = [['Average', mcc_hypo_avg_str, mcc_hyper_avg_str, total_avg_str]]
    table_data += new_row
    c = draw_table(c, table_data, 700 - 20 * int(df['prediction_horizon'][0]) // table_interval)
    return c


def draw_physiological_alignment_table(c, df, feature, y_placement):
    sign = 'Sign'
    if feature == 'carbs':
        sign = 'Positive Sign'
    else:
        sign = 'Negative Sign'

    table_data = [
        [sign, f'Persistence', f'Total']
    ]

    prediction_horizon = get_ph(df)
    partial_dependencies_str = df['partial_dependencies'][0]
    partial_dependencies_dict = ast.literal_eval(partial_dependencies_str)

    # Add column time lags to a list
    time_lags = []
    for column_title, _ in partial_dependencies_dict.items():
        if column_title == feature:
            time_lags += [0]
        elif column_title.startswith(f'{feature}_what_if'):
            time_lag = int(column_title.split('_')[-1])
            time_lags += [time_lag]
        elif column_title.startswith(f'{feature}'):
            time_lag = -int(column_title.split('_')[-1])
            time_lags += [time_lag]

    pd_numbers = []
    for time_lag in time_lags:
        if time_lag < 0:
            pd_numbers += [partial_dependencies_dict[f'{feature}_{np.abs(time_lag)}']]
        elif time_lag == 0:
            pd_numbers += [partial_dependencies_dict[f'{feature}']]
        else:
            new_row = partial_dependencies_dict[f'{feature}_what_if_{np.abs(time_lag)}']
            prediction_index = time_lag // 5 - 1
            new_row[0:prediction_index] = [np.nan] * prediction_index
            pd_numbers += [new_row]

    total_values = 0
    correct_sign_values = 0
    persistant_values = 0

    for row in pd_numbers:
        prev_value = 0
        for val in row:
            if not np.isnan(val):
                total_values += 1
                if feature == 'carbs':
                    if val > 0:
                        correct_sign_values += 1
                    if val > prev_value:
                        persistant_values += 1
                else:
                    if val < 0:
                        correct_sign_values += 1
                    if val < prev_value:
                        persistant_values += 1
                prev_value = val

    percentage_below_zero = (correct_sign_values / total_values) * 100
    percentage_persistant_values = (persistant_values / total_values) * 100
    total = (percentage_below_zero + percentage_persistant_values) / 2
    table_data += [[f'{int(percentage_below_zero)}%', f'{int(percentage_persistant_values)}%', f'{int(total)}%']]
    c = draw_table(c, table_data, y_placement)
    return c


def draw_physiological_alignment_single_dimension_table(c, df, feature, y_placement):
    if 'daily_avg_insulin' in df.columns:
        if feature == 'carbs':
            sign = 'Positive Sign'
        else:
            sign = 'Negative Sign'

        # TODO: Add evaluation of magnitude
        if feature == 'carbs':
            table_data = [
                [sign, f'Persistence', 'Carb ratio (CR)', 'Expected CR [g/U]', f'Total']
            ]
            expected_ratio = 500 / df['daily_avg_insulin'][0]
        else:
            table_data = [
                [sign, f'Persistence', 'Insulin sensitivity factor (ISF)',
                 f'Expected ISF [{unit_config_manager.get_unit()}/U]', f'Total']
            ]
            expected_ratio = 1800 / df['daily_avg_insulin'][0]

        col_prefix = f'partial_dependency_{feature}_'
        columns = [column for column in df.columns if column.startswith(col_prefix)]

        quantities = []
        pd_numbers = []

        for column in columns:
            y_pred = df[column][0]
            y_pred = y_pred.replace("nan", "None")
            y_pred = ast.literal_eval(y_pred)
            quantities += [float(column.split('_')[-1])]
            pd_numbers += [y_pred]

        total_values = 0
        correct_sign_values = 0
        persistant_values = 0

        estimated_ISF = 1800 / df['daily_avg_insulin'][0]
        calculated_ratios = []
        for index, row in enumerate(pd_numbers):
            prev_value = row[0]

            for i in range(1, len(row)):
                if not np.isnan(row[i]):
                    total_values += 1
                    if feature == 'carbs':
                        if row[i] > 0:
                            correct_sign_values += 1
                        if row[i] > prev_value:
                            persistant_values += 1

                        if not row[i] == 0:
                            calculated_CR = quantities[index] * get_fraction_absorbed((i + 1) * 5, 60,
                                                                                      180) * estimated_ISF / \
                                            row[i]
                            calculated_ratios += [calculated_CR]
                        else:
                            calculated_ratios += [np.nan]
                    else:
                        if row[i] < 0:
                            correct_sign_values += 1
                        if row[i] < prev_value:
                            persistant_values += 1
                        calculated_ISF = row[i] / (quantities[index] * get_fraction_absorbed((i + 1) * 5, 75, 300))
                        calculated_ratios += [-calculated_ISF]
                    prev_value = row[i]

        percentage_below_zero = (correct_sign_values / total_values) * 100
        percentage_persistant_values = (persistant_values / total_values) * 100
        total = (percentage_below_zero + percentage_persistant_values) / 2
        table_data += [[f'{int(percentage_below_zero)}%', f'{int(percentage_persistant_values)}%',
                        "{:.1f}".format(np.mean(calculated_ratios)), "{:.1f}".format(expected_ratio), f'{int(total)}%']]
        c = draw_table(c, table_data, y_placement)
        return c
    else:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, y_placement, f'The model does not use {feature} as an input.')
        return c


def draw_table(c, data, y_placement):
    table = Table(data)
    table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ('GRID', (0, 1), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0, colors.black)
    ]))

    # Draw the table on the canvas
    table.wrapOn(c, 0, 0)
    table.drawOn(c, 100, y_placement)
    return c


def plot_across_prediction_horizons(c, df, title, columns, height=2, y_labels=None, y_label='', y_placement=300):
    x_values = list(range(5, get_ph(df) + 1, 5))
    fig = plt.figure(figsize=(5.5, height))

    for i, val in enumerate(columns):
        values = []
        for ph in x_values:
            values += [float(df[f'{val}_{ph}'][0])]

        if y_labels:
            plt.plot(x_values, values, marker='o', label=y_labels[i])
        else:
            plt.plot(x_values, values, marker='o')

    # Setting the title and labels with placeholders for the metric unit
    plt.title(title)
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(y_label)
    if y_labels:
        plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, y_placement)

    plt.close(fig)
    return c


def plot_rmse_across_prediction_horizons(c, dfs, height=2, y_placement=300):
    fig = plt.figure(figsize=(5.5, height))

    for df in dfs:
        model_name = df['Model Name'][0]
        x_values = list(range(5, get_ph(df) + 1, 5))
        y_values = []
        for ph in x_values:
            y_values += [float(df[f'rmse_{ph}'][0])]

        plt.plot(x_values, y_values, marker='o', label=model_name)

    # Setting the title and labels with placeholders for the metric unit
    plt.title('Model Accuracy in RMSE')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'RMSE [{unit_config_manager.get_unit()}]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, y_placement)

    plt.close(fig)
    return c


def plot_error_grid_across_prediction_horizons(c, dfs, height=2, y_placement=300):
    fig = plt.figure(figsize=(5.5, height))

    for df in dfs:
        model_name = df['Model Name'][0]
        x_values = list(range(5, get_ph(df) + 1, 5))
        y_values = []
        for ph in x_values:
            y_values += [float(df[f'parkes_error_grid_exp_{ph}'][0] * 100)]

        plt.plot(x_values, y_values, marker='o', label=model_name)

    # Setting the title and labels with placeholders for the metric unit
    plt.title('Model Accuracy in Pakes Error Grid Analysis')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Pakes Error Grid Analysis [%]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, y_placement)

    plt.close(fig)
    return c


def plot_mcc_across_prediction_horizons(c, dfs, height=2, y_placement=300):
    fig = plt.figure(figsize=(5.5, height))

    for df in dfs:
        model_name = df['Model Name'][0]
        x_values = list(range(5, get_ph(df) + 1, 5))
        y_values = []
        for ph in x_values:
            y_values += [float((df[f'mcc_hypo_{ph}'][0] + df[f'mcc_hyper_{ph}'][0]) / 2)]
        plt.plot(x_values, y_values, marker='o', label=model_name)

    # Setting the title and labels with placeholders for the metric unit
    plt.title('Glycemia Detection - MCC')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Average MCC')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, y_placement)

    plt.close(fig)
    return c


def plot_partial_dependencies_across_prediction_horizons(c, df, col, height=2, y_placement=300):
    fig = plt.figure(figsize=(5.5, height))
    col_prefix = f'partial_dependency_{col}_'
    columns = [column for column in df.columns if column.startswith(col_prefix)]

    if not columns:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, y_placement, f'The model does not use {col} as an input.')
        return c
    else:
        x_values = list(range(5, get_ph(df) + 1, 5))

        for column in columns:
            quantity = column.split('_')[-1]
            if col == 'carbs':
                label = f'{quantity}g of carbohydrates'
            else:
                label = f'{quantity}U of insulin'
            y_pred = df[column][0]
            y_pred = ast.literal_eval(y_pred)
            plt.plot(x_values, y_pred, marker='o', label=label)

        # Setting the title and labels with placeholders for the metric unit
        plt.title(f'Partial Dependency Plot for {col}')
        plt.xlabel('Prediction Horizons (minutes)')
        plt.ylabel(f'Change in predicted blood glucose [{unit_config_manager.get_unit()}]')
        plt.legend()

        # Save the plot as an image
        buffer = BytesIO()
        fig.savefig(buffer, format='svg')

        buffer.seek(0)  # Move the file pointer to the beginning
        drawing = svg2rlg(buffer)

        renderPDF.draw(drawing, c, 70, y_placement)

        plt.close(fig)
        return c


def plot_predicted_dristribution_across_prediction_horizons(c, dfs, height=2, y_placement=300):
    fig = plt.figure(figsize=(5.5, height))

    for df in dfs:
        model_name = df['Model Name'][0]
        x_values = list(range(5, get_ph(df) + 1, 5))
        y_values = []
        for ph in x_values:
            y_test = df[f'target_{ph}'][0]
            y_pred = df[f'y_pred_{ph}'][0].replace("nan", "None")
            y_test = ast.literal_eval(y_test)
            y_pred = ast.literal_eval(y_pred)
            y_pred = [np.nan if x is None else x for x in y_pred]
            y_values += [np.nanstd(y_pred) / np.nanstd(y_test) * 100]
        plt.plot(x_values, y_values, marker='o', label=model_name)

    # Setting the title and labels with placeholders for the metric unit
    plt.title('Standard Deviation')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Standard Deviation of Predicted versus Measured [%]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, y_placement)

    plt.close(fig)
    return c


def draw_scatter_plot(c, df, ph, x_placement, y_placement):
    fig = plt.figure(figsize=(2, 2))

    y_test = df[f'target_{ph}'][0]
    y_pred = df[f'y_pred_{ph}'][0]

    y_test = ast.literal_eval(y_test)
    y_pred = y_pred.replace("nan", "None")
    y_pred = ast.literal_eval(y_pred)

    plt.scatter(y_test, y_pred, alpha=0.5)

    if unit_config_manager.use_mgdl:
        unit = "mg/dL"
        max_val = 400
    else:
        unit = "mmol/L"
        max_val = unit_config_manager.convert_value(400)

    # Plotting the line x=y
    plt.plot([0, max_val], [0, max_val], 'k-')

    plt.xlabel(f"True Blood Glucose [{unit}]")
    plt.ylabel(f"Predicted Blood Glucose [{unit}]")
    plt.title(f"{ph} minutes")

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, x_placement, y_placement)

    plt.close(fig)
    return c


def plot_predicted_distribution(c, df, x_placement, y_placement):
    fig = plt.figure(figsize=(5, 3))
    prediction_horizon = get_ph(df)
    x_values = list(range(5, prediction_horizon + 1, 5))

    y_pred_std = []
    y_test_std = []

    for ph in x_values:
        y_test = df[f'target_{ph}'][0]
        y_pred = df[f'y_pred_{ph}'][0]
        y_test = ast.literal_eval(y_test)
        y_pred = y_pred.replace("nan", "None")
        y_pred = ast.literal_eval(y_pred)
        y_pred = [np.nan if x is None else x for x in y_pred]
        y_pred_std += [np.nanstd(y_pred)]
        y_test_std += [np.nanstd(y_test)]

    plt.plot(x_values, y_pred_std, marker='o', label=f'Predicted Values')
    plt.plot(x_values, y_test_std, marker='o', label=f'Measured Values')

    plt.title(f'Prediction Standard Deviation')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Standard Deviation [{unit_config_manager.get_unit()}]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, x_placement, y_placement)

    plt.close(fig)
    return c


def plot_confusion_matrix(c, df, classes, ph, x_placement, y_placement, cmap=plt.cm.Blues):
    percentages = df[f'glycemia_detection_{ph}'][0]
    percentages = ast.literal_eval(percentages)

    fig = plt.figure(figsize=(3, 2.5))
    sns.heatmap(percentages, annot=True, cmap=cmap, fmt='.2%', xticklabels=classes, yticklabels=classes)
    plt.title(f'{ph} minutes')
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, x_placement, y_placement)

    plt.close(fig)
    return c


def plot_partial_dependency_heatmap(c, df, feature, x_placement, y_placement, title):
    prediction_horizon = get_ph(df)
    partial_dependencies_str = df['partial_dependencies'][0]
    partial_dependencies_dict = ast.literal_eval(partial_dependencies_str)

    # Define custom colormap
    colors = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]  # Red, White, Green
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Add column time lags to a list
    y_tick_labels = []
    for column_title, _ in partial_dependencies_dict.items():
        if column_title == feature:
            y_tick_labels += [0]
        elif column_title.startswith(f'{feature}_what_if'):
            time_lag = int(column_title.split('_')[-1])
            y_tick_labels += [time_lag]
        elif column_title.startswith(f'{feature}'):
            time_lag = -int(column_title.split('_')[-1])
            y_tick_labels += [time_lag]

    interval = 30
    y_tick_labels = sorted(y_tick_labels)
    y_tick_labels = [num for num in y_tick_labels if num % interval == 0]
    x_tick_labels = range(5, prediction_horizon + 1, 5)

    pd_numbers = []
    for time_lag in y_tick_labels:
        if time_lag < 0:
            pd_numbers += [partial_dependencies_dict[f'{feature}_{np.abs(time_lag)}']]
        elif time_lag == 0:
            pd_numbers += [partial_dependencies_dict[f'{feature}']]
        else:
            pd_numbers += [partial_dependencies_dict[f'{feature}_what_if_{np.abs(time_lag)}']]

    # Determine the maximum absolute value in the partial dependency numbers
    max_abs_pd = max(abs(val) for row in pd_numbers for val in row)

    # Set the bounds for the colormap symmetrically around 0
    bounds = np.linspace(-max_abs_pd, max_abs_pd, 101)  # Adjust the number of levels as needed
    if max_abs_pd == 0:
        bounds = np.linspace(-0.01, 0.01, 101)  # Small range around zero
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(5, 2.5))
    ax = sns.heatmap(pd_numbers, annot=False, cmap=cmap, norm=norm, xticklabels=x_tick_labels,
                     yticklabels=y_tick_labels)

    n = len(x_tick_labels)
    n_rows = len(pd_numbers) * (interval // 5)
    for i in range(n):
        # Add a grey zone under the diagonal
        plt.axvspan(0, i + 1, ymin=0, ymax=(n - (i + 1)) / (n_rows - 1), color='grey', alpha=0.1)

    # Manually set some x-axis tick labels to empty strings
    num_ticks = len(ax.get_xticks())
    x_tick_labels = [x_tick_labels[i] if (i) % (len(x_tick_labels) // 18) == 0 else '' for i in range(num_ticks)]
    ax.set_xticklabels(x_tick_labels)

    plt.title(title)
    plt.ylabel(f'Time lag for {feature} (minutes)')
    plt.xlabel('Prediction Horizon (minutes)')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, x_placement, y_placement)

    plt.close(fig)
    return c


def get_fraction_absorbed(t, peak, total_activity_time):
    """
    Calculate the area under the curve (AUC) for the absorption of a substance from time 0 to t,
    with a given peak and total activity duration.

    Parameters:
    - t (float): The current time up to which the AUC is calculated. Must be between 0 and total_activity_time.
    - peak (float): The time at which the peak absorption occurs.
    - total_activity_time (float): The total time over which the substance is active.

    Returns:
    - float: The fraction of the AUC from time 0 to t.
    """
    total_auc = total_activity_time / 2

    if t < 0:
        return 0  # No absorption outside the activity time range.

    if t <= peak:
        # AUC of a triangle from 0 to t (linear increase)
        height = (1 / peak) * t
        auc = 0.5 * t * height / total_auc
    else:
        # Full AUC from 0 to peak
        full_peak_auc = 0.5 * 1 * peak
        full_post_peak_auc = 0.5 * 1 * (total_activity_time - peak)

        if t < total_activity_time:
            # AUC from peak to t (linear decrease)
            # Base = t - peak, Height starts at 100 and goes to ((total_activity_time - t) / (total_activity_time - peak)) * 100
            height = ((total_activity_time - t) / (total_activity_time - peak))
            post_peak_auc = full_post_peak_auc - (0.5 * height * (total_activity_time - t))
            auc = (full_peak_auc + post_peak_auc) / total_auc
        else:
            # Full AUC from peak to total_activity_time
            auc = 1

    return auc


def get_ph(df):
    return int(df['prediction_horizon'][0])
