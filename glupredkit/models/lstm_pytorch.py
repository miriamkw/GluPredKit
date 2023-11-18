import torch
import torch.nn as nn
import torch.optim as optim
from .base_model import BaseModel
import numpy as np
import os
from datetime import datetime
import ast
from torch.utils.data import DataLoader, TensorDataset
from glupredkit.helpers.tf_keras import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.num_inputs = None

        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.pytorch_models/lstm_pytorch_ph-{prediction_horizon}_{safe_timestamp}.pth"

        # Extract the directory path
        model_dir = os.path.dirname(self.model_path)
        # Check if the directory exists, and if not, create it
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def fit(self, x_train, y_train):
        # Convert the 'sequence' column from strings to NumPy arrays
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        sequences = np.array(sequences)

        # Convert targets to NumPy array
        targets = np.array(y_train.tolist())

        # Define the split index for training and validation sets
        split_idx = int(len(sequences) * 0.8)  # 80% for training, 20% for validation

        # Split sequences and targets into training and validation sets
        train_sequences, val_sequences = sequences[:split_idx], sequences[split_idx:]
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]

        # Convert training data to PyTorch tensors and create DataLoader
        train_sequences_tensor = torch.from_numpy(train_sequences).float()
        train_targets_tensor = torch.from_numpy(train_targets).float()
        train_loader = DataLoader(TensorDataset(train_sequences_tensor, train_targets_tensor), batch_size=1,
                                  shuffle=True)

        # Convert validation data to PyTorch tensors and create DataLoader
        val_sequences_tensor = torch.from_numpy(val_sequences).float()
        val_targets_tensor = torch.from_numpy(val_targets).float()
        val_loader = DataLoader(TensorDataset(val_sequences_tensor, val_targets_tensor), batch_size=1, shuffle=False)

        # Define model
        self.num_inputs = sequences.shape[2]  # Number of features
        model = self.get_model(num_of_inputs=self.num_inputs)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        num_epochs = 20
        best_val_loss = float('inf')
        print(f'Starting first of {num_epochs} epochs...')

        # Training loop
        for epoch in range(num_epochs):
            #  Training Phase
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            #  Validation Phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)

            # Update scheduler
            if epoch >= 10:  # Start reducing LR after 10 epochs
                scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.model_path)

            print(f'Epoch [{epoch + 1}/{num_epochs}] finished')

        return self

    def predict(self, x_test):
        model = self.get_model(self.num_inputs)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)
        inputs = torch.from_numpy(sequences).float()
        with torch.no_grad():
            predictions = model(inputs)
        return predictions.numpy()

    def best_params(self):
        # Implement as needed, possibly using a hyperparameter tuning method
        return None

    def process_data(self, df, model_config_manager, real_time):
        # Implement preprocessing specific to your TCN and dataset
        return process_data(df, model_config_manager, real_time)

    def get_model(self, num_of_inputs):
        return LSTMNetwork(num_of_inputs)


class LSTMNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size=50, num_layers=3):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)  # Assuming a single output value per sequence

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        return self.linear(last_time_step_out).squeeze()

