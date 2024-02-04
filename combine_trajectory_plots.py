from PIL import Image, ImageDraw, ImageFont
import os

# Path to the folder containing subfolders with plot PNG files
folder_path = r'C:\Users\DoyoungOh\GluPredKit\data\figures'

# List of models
models = ["arx", "double_lstm", "elastic_net", "huber", "lasso", "lstm", "lstm_pytorch", "my_lstm", "my_mlp", "plsr_with_diff_data_process", "random_forest", "ridge", "stacked_mlp_and_plsr", "stacked_with_plsr", "svr_linear", "svr_rbf", "tcn_pytorch", "tcn"]
model_images = []

# Loop through each model
for model in models:
    # Create a list to store image objects for each row
    row_images = []
    model_folder = os.path.join(folder_path, model)
    plot_files = [file for file in os.listdir(model_folder) if file.endswith('.png')]
    plot_files.sort()

    # Loop through the plot files, 4 plots at a time
    for i in range(0, len(plot_files), 4):
        column_images = []

        for j in range(i, min(i + 4, len(plot_files))):
            # Open the plot PNG file
            plot_path = os.path.join(model_folder, plot_files[j])
            image = Image.open(plot_path)

            # Resize images to maintain aspect ratio
            # aspect_ratio = image.width / image.height
            # new_width = 300  # Adjust as needed
            # new_height = int(new_width / aspect_ratio)
            # image = image.resize((new_width, new_height))

            # Append the image to the column list
            column_images.append(image)

        # Combine images horizontally for the current row
        row_image = Image.new('RGB', (sum(img.width for img in column_images), column_images[0].height))
        x_offset = 0
        for img in column_images:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width
        row_images.append(row_image)

    # Combine row images vertically for the current model
    combined_row_image = Image.new('RGB', (row_images[0].width, sum(img.height for img in row_images)))
    y_offset = 0
    for img in row_images:
        combined_row_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Add the combined row image to the list of model images with model name
    model_images.append((model, combined_row_image))

# Combine all model images vertically and mark each model
total_width = max(img.width for (_, img) in model_images)
combined_height = sum(img.height for (_, img) in model_images)
combined_image = Image.new('RGB', (total_width, combined_height), color='white')
draw = ImageDraw.Draw(combined_image)
font = ImageFont.truetype("arial.ttf", size=35)  # You can adjust the font as needed

y_offset = 0
for model, img in model_images:
    combined_image.paste(img, (0, y_offset))
    draw.text((10, y_offset + 10), model, fill='black', font=font)
    y_offset += img.height

# Save the combined image
combined_image_path = os.path.join(folder_path, 'combined_models.png')
combined_image.save(combined_image_path)

print("Combined images saved successfully.")