import subprocess

model_names = ["arx", "double_lstm", "elastic_net", "huber", "lasso", "lstm", "lstm_pytorch", "my_lstm", "my_GBT", "my_mlp", "plsr_with_diff_data_process", "random_forest", "ridge", "stacked_mlp_and_plsr", "stacked_with_plsr", "svr_linear", "svr_rbf", "tcn_pytorch", "tcn"]
# ph_values= [30, 60]
ph_values= [60]

# Loop through each model name and pH value
for model_name in model_names:
    for ph in ph_values:
        # Construct the command string
        command = f"python -m glupredkit.cli draw_plots --models {model_name}__train_config__{ph}.pkl --plots trajectories"
        
        # Execute the command using subprocess
        subprocess.run(command, shell=True)
