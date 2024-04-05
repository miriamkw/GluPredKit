import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from glupredkit.helpers.unit_config_manager import unit_config_manager
from io import BytesIO
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg


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

    # Normal text
    canvas.setFont("Helvetica", 12)
    canvas.drawString(100, 540, f'Dataset: {df["data"][0]}')
    canvas.drawString(100, 520, f'Data ids: {df["subject_ids"][0]}')
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


def set_bottom_text(canvas):
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(letter[0] / 2, 50, f'This report is generated using GluPredKit '
                                                f'(https://github.com/miriamkw/GluPredKit)')
    return canvas



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
            values += [float(df[f'{val}_{ph}'])]

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
    return c


def get_ph(df):
    return int(df['prediction_horizon'][0])

