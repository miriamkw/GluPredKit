import subprocess

# "arx", "double_lstm", "elastic_net", "huber", "lasso", "lstm", "lstm_pytorch", "my_GBT", "my_lstm", "my_mlp", "plsr_with_diff_data_process", "random_forest", "ridge", "stacked_mlp_and_plsr", "stacked_with_plsr", 
# List of model names
model_names = [ "svr_linear", "svr_rbf", "tcn_pytorch", "tcn"]


# Loop through each model name and pH value
for model_name in model_names:
    # Construct the command string
    command = f"python -m glupredkit.cli train_model {model_name} test_config"
    
    # Execute the command using subprocess
    subprocess.run(command, shell=True)
