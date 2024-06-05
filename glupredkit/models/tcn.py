import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from .base_model import BaseModel
import numpy as np
import os
from datetime import datetime
import ast
from torch.utils.data import DataLoader, TensorDataset, Dataset
from glupredkit.helpers.tf_keras import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.num_inputs = None
        self.num_outputs = None
        self.n_channels = None
        self.kernel_size = 5
        self.dropout = 0.25

        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        safe_timestamp = safe_timestamp.replace('.', '_')
        self.model_path = f"data/.pytorch_models/tcn_pytorch_ph-{prediction_horizon}_{safe_timestamp}.pth"

        # Extract the directory path
        model_dir = os.path.dirname(self.model_path)
        # Check if the directory exists, and if not, create it
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def _fit_model(self, x_train, y_train, epochs=20, *args):
        # Convert the 'sequence' column from strings to NumPy arrays
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_train['target']]

        sequences = np.array(sequences)
        targets = np.array(targets)

        dataset = TimeSeriesDataset(sequences, targets)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        # Define the model
        self.num_inputs = sequences.shape[2]  # Number of features
        self.num_outputs = targets.shape[1]
        self.n_channels = [150] * 4
        model = TCN(input_size=self.num_inputs, output_size=self.num_outputs, num_channels=self.n_channels,
                    kernel_size=self.kernel_size, dropout=self.dropout)
        model.double()  # Ensure the model matches the double precision of the targets

        # Define loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        print(f'Starting first of {epochs} epochs...')

        # Training loop
        for epoch in range(epochs):
            #  Training Phase
            model.train()
            total_loss = 0

            for inputs, targets in dataloader:
                inputs = inputs.double()  # Convert inputs to float32
                targets = targets.double()  # Convert targets to float32

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
        torch.save(model.state_dict(), self.model_path)
        return self

    def _predict_model(self, x_test):
        model = TCN(input_size=self.num_inputs, output_size=self.num_outputs, num_channels=self.n_channels,
                    kernel_size=self.kernel_size, dropout=self.dropout)
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


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target

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

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        # output = self.linear(output).double()
        last_step_output = output[:, -1, :]
        output = self.linear(last_step_output).double()
        return self.sig(output)


