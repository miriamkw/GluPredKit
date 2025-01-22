from glupredkit.helpers.scikit_learn import process_data
from .base_model import BaseModel
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.input_dim = None
        self.output_dim = None
        self.alpha = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Simple linear model with multiple outputs
        self.model = None
        #self.criterion = nn.MSELoss()
        self.criterion = CustomLoss(alpha=0.1)

        self.scaler = None
        self.y_scaler = None


    def _fit_model(self, x_train, y_train, epochs=1000, *args):

        # TODO: I think the final solution will be to train separate models for each prediction horizon!

        x_train.drop(['id'], axis=1, inplace=True)
        learning_rate = 0.01

        self.input_dim = len(x_train.columns)
        self.output_dim = len(y_train.columns)
        self.model = nn.Linear(self.input_dim, self.output_dim).to(self.device)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        self.scaler = scaler

        # Dataframe with delta bg
        df_delta_bg = y_train.diff().fillna(0.0)

        # Convert DataFrames to lists of lists and compute weights
        y_train_list = y_train.to_numpy().tolist()
        df_delta_bg_list = df_delta_bg.to_numpy().tolist()
        weights = [
            [self.get_weight(bg, delta_bg) for bg, delta_bg in zip(row_y, row_delta)]
            for row_y, row_delta in zip(y_train_list, df_delta_bg_list)
        ]

        #weights = y_train.applymap(self.get_weight)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.values)
        self.y_scaler = y_scaler

        # Convert DataFrame to PyTorch tensors
        X_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)

        # Optimizer with L2 penalty (Ridge regularization)
        # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=self.alpha)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) #, weight_decay=self.alpha)

        for epoch in range(epochs):
            self.model.train()

            # Forward pass
            y_pred = self.model(X_train)
            loss = self.criterion.forward(y_pred, y_train, weights)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        return self

    def _predict_model(self, x_test):
        test_data = x_test.copy()
        test_data.drop(['id'], axis=1, inplace=True)
        test_data = self.scaler.transform(test_data)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_data)
            predictions = self.y_scaler.inverse_transform(predictions.cpu().detach().numpy())

        return predictions

    def best_params(self):
        return ''

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)


    def get_weight(self, bg, delta_bg):

        # Add one so that the lowest cost is 1 instead of 0
        # (we don't want to cancel out prediction errors of low cost predictions!)
        weight = self.zone_cost(bg) + self.slope_cost(bg, delta_bg) + 1
        return weight

    def zone_cost(self, bg, target=105):
        if bg < 1:
            bg = 1
        if bg > 600:
            bg = 600

        # This function assumes BG in mg / dL
        constant = 32.9170208165394
        left_weight = 40.0
        right_weight = 1.7

        if bg < target:
            risk = constant * left_weight * (np.log(bg) - np.log(target)) ** 2
        else:
            risk = constant * right_weight * (np.log(bg) - np.log(target)) ** 2

        return risk


    def slope_cost(self, bg, delta_bg):

        # TODO: Might tune this, especially since the zone cost is adjusted!

        k = 18.0182
        # This function assumes mmol/L
        bg = bg / k
        delta_bg = delta_bg / k

        a = bg
        b = 15 - bg
        if b < 0:
            b = 0
        if a > 15:
            a = 15

        cost = (np.sign(delta_bg) + 1) / 2 * a * (delta_bg ** 2) - 2 * (np.sign(delta_bg) - 1) / 2 * b * (delta_bg ** 2)
        return cost


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.1, target=105):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.target = target

    def forward(self, y_pred, y_true, weights):

        # TODO: Add slope cost as well - not just zone cost!!

        mse_loss = torch.mean(weights * (y_pred - y_true) ** 2)  # Mean Squared Error
        l1_loss = torch.mean(torch.abs(y_pred - y_true))  # Mean Absolute Error
        return mse_loss + self.alpha * l1_loss



