from glupredkit.helpers.scikit_learn import process_data
from .base_model import BaseModel
from sklearn.preprocessing import StandardScaler
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
        self.criterion = nn.MSELoss()

        self.scaler = None
        self.y_scaler = None


    def _fit_model(self, x_train, y_train, epochs=500, *args):

        # TODO: I think the final solution will be to train separate models for each prediction horizon!

        x_train.drop(['id'], axis=1, inplace=True)
        learning_rate = 0.01

        self.input_dim = len(x_train.columns)
        self.output_dim = len(y_train.columns)
        self.model = nn.Linear(self.input_dim, self.output_dim).to(self.device)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        self.scaler = scaler

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
            loss = self.criterion(y_pred, y_train)

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

