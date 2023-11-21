import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
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
        self.model_path = f"data/.pytorch_models/tcn_pytorch_ph-{prediction_horizon}_{safe_timestamp}.pth"

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
        train_sequences_tensor = torch.from_numpy(train_sequences).float().transpose(1, 2)
        train_targets_tensor = torch.from_numpy(train_targets).float()
        train_loader = DataLoader(TensorDataset(train_sequences_tensor, train_targets_tensor), batch_size=1,
                                  shuffle=True)

        # Convert validation data to PyTorch tensors and create DataLoader
        val_sequences_tensor = torch.from_numpy(val_sequences).float().transpose(1, 2)
        val_targets_tensor = torch.from_numpy(val_targets).float()
        val_loader = DataLoader(TensorDataset(val_sequences_tensor, val_targets_tensor), batch_size=1,
                                shuffle=False)

        # Define the model
        self.num_inputs = sequences.shape[2]  # Number of features
        model = self.get_model(num_of_inputs=self.num_inputs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.MSELoss()
        # Initialize ReduceLROnPlateau scheduler
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

                # Reshape the outputs and targets to be consistent
                outputs = outputs.view(-1)  # Ensures outputs is a 1D tensor
                targets = targets.view(-1)  # Ensures targets is a 1D tensor
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
                    outputs = outputs.view(-1)
                    targets = targets.view(-1)
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
        inputs = torch.from_numpy(sequences).float().transpose(1, 2)
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
        # Defining settings like in the benchmark paper
        num_channels = [50] * 10  # 10 layers with the same number of filters
        kernel_size = 4
        dropout = 0.5
        return TemporalConvNet(num_inputs=num_of_inputs, num_channels=num_channels, kernel_size=kernel_size,
                               dropout=dropout)


"""
The rest of this file contains code adapted from Shaojie Bai, J. Zico Kolter and Vladlen Koltun's Sequence Modeling 
Benchmarks and Temporal Convolutional Networks (TCN) at https://github.com/locuslab/TCN/tree/master.
The original code is licensed under the MIT License. The license text is as follows:

MIT License

Copyright (c) 2018 CMU Locus Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = self.network(x)
        x = x[:, :, -1]  # Use the output of the last time step
        x = self.linear(x)  # Apply the linear layer
        return x.squeeze()  # Remove extra dimensions for single value prediction

