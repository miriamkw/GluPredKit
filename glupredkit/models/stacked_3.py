from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
import ast
import numpy as np
import tensorflow as tf
from datetime import datetime
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error

# SImple stacking (averaging the prediction)
# Ohio
# 17.21, 29.08
# Participant 1
# 17.65, 26.31
# Participant 2
# 22.49, 36.93
class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/stacked_tcn_ph-{prediction_horizon}_{safe_timestamp}.keras"
        self.tcn_model = None
        self.mlp_model = None
        self.plsr_model = None
        self.ridge_model = None
        # self.stacked_model = None

    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()
        x_train = np.array(sequences)
        y_train = np.array(targets)

        # TCN
        self.tcn_model = self._train_tcn(x_train, y_train)

        # Regression models
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.ridge_model = RidgeCV()
        self.mlp_model = self._create_mlp()  
        self.plsr_model = self._create_plsr(x_train, y_train)
        self.ridge_model.fit(x_train, y_train)
        self.mlp_model.fit(x_train, y_train)
        self.plsr_model.fit(x_train, y_train)

        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        self.tcn_model = self.get_model(self.num_inputs)
        self.tcn_model.load_state_dict(torch.load(self.model_path))
        self.tcn_model.eval()
        inputs = torch.from_numpy(sequences).float().transpose(1, 2)
        with torch.no_grad():
            predictions = self.tcn_model(inputs)
        pred_tcn = predictions.numpy()

        sequences = sequences.reshape(sequences.shape[0], -1)
        pred_ridge = self.ridge_model.predict(sequences)
        pred_mlp = self.mlp_model.predict(sequences)
        pred_plsr = self.plsr_model.predict(sequences)

        predictions = []
        predictions.append(pred_tcn)
        predictions.append(pred_ridge)
        predictions.append(pred_mlp)
        predictions.append(pred_plsr)

        return np.mean(predictions, axis=0)
        #return [val[0] for val in predictions]

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    
    def get_model(self, num_of_inputs):
        # Defining settings like in the benchmark paper
        num_channels = [50] * 10  # 10 layers with the same number of filters
        kernel_size = 4
        dropout = 0.5
        return TemporalConvNet(num_inputs=num_of_inputs, num_channels=num_channels, kernel_size=kernel_size,
                               dropout=dropout)
    
    def _train_tcn(self, sequences, targets):
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
    

    def _create_mlp(self):
        return MLPRegressor(hidden_layer_sizes=(40,40), max_iter=500, activation='relu', 
                            solver='adam', random_state=42, shuffle=False, 
                            early_stopping=True)
    

    def _create_plsr(self, X_train, Y_train):
        # maximum number of components based on the number of features
        componentmax = X_train.shape[1]
        component = np.arange(1, componentmax)
        rmse = []

        # loop over different numbers of components to find the best amount of components
        for i in component:
            pls = PLSRegression(n_components=i)
            pls.fit(X_train, Y_train)
            Y_pred_train = pls.predict(X_train)
            msecv = mean_squared_error(Y_train, Y_pred_train)
            rmsecv = np.sqrt(msecv)
            rmse.append(rmsecv)

        # find the number of components that minimizes RMSE
        msemin = np.argmin(rmse)
        # set the optimal number of components
        n_components = msemin + 1
        # create a new PLS Regression model with the optimal number of components
        return PLSRegression(n_components)
    


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
