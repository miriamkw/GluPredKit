"""
Stacked PLSR, from: https://ceur-ws.org/Vol-2675/paper21.pdf
GitHub: https://gitlab.com/Hoda-Nemat/data-fusion-stacking
"""
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from glupredkit.helpers.tf_keras import process_data
from .base_model import BaseModel
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
from sklearn.cross_decomposition import PLSRegression
from keras.layers import LSTM, Dense


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')
        # models
        self.mlp_model = None
        self.first_plsr_model = None
        self.second_plsr_model = None
        self.lstm_model_path = f"data/.keras_models/stacked_plsr_ph-{prediction_horizon}_{safe_timestamp}.h5"
        self.stacked_model = None

    def _fit_model(self, x_train, y_train, *args):
        x_train = x_train['sequence'].apply(ast.literal_eval)
        x_train = np.array(x_train.tolist())
        x_train_flat = x_train.reshape(x_train.shape[0], -1)  # Flatten each sample

        y_train = [np.array(ast.literal_eval(target_str)) for target_str in y_train['target']]
        y_train = np.array(y_train)

        n_components = self._get_plsr_components(x_train_flat, y_train)
        self.first_plsr_model = PLSRegression(n_components)
        self.mlp_model = self._create_mlp()
        lstm_model = self._create_lstm(x_train)

        self.mlp_model.fit(x_train_flat, y_train)
        self.first_plsr_model.fit(x_train_flat, y_train)
        lstm_model.fit(x_train, y_train)
        lstm_model.save(self.lstm_model_path)

        first_level_1 = self.mlp_model.predict(x_train_flat)
        first_level_2 = self.first_plsr_model.predict(x_train_flat)
        first_level_3 = lstm_model.predict(x_train)

        # n_components = self._get_plsr_components(np.column_stack((first_level_1, first_level_2, first_level_3)), y_train)
        # print("plsr components:", n_components)
        self.second_plsr_model = PLSRegression(3)
        self.second_plsr_model.fit(np.column_stack((first_level_1, first_level_2, first_level_3)), y_train)

        return self

    def _predict_model(self, x_test):
        x_test = x_test['sequence'].apply(ast.literal_eval)
        x_test = np.array(x_test.tolist())
        x_test_flat = x_test.reshape(x_test.shape[0], -1)

        lstm_model = tf.keras.models.load_model(self.lstm_model_path,
                                                custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})

        first_level_1 = self.mlp_model.predict(x_test_flat)
        first_level_2 = self.first_plsr_model.predict(x_test_flat)
        first_level_3 = lstm_model.predict(x_test)

        first_level_pred = np.column_stack((first_level_1, first_level_2, first_level_3))
        second_level_pred = self.second_plsr_model.predict(first_level_pred)
        return second_level_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def _create_mlp(self):
        return MLPRegressor(hidden_layer_sizes=(40, 40), max_iter=500, activation='relu',
                            solver='adam', random_state=42, shuffle=False,
                            early_stopping=True)

    def _get_plsr_components(self, X_train, Y_train):
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
        return n_components

    def _create_lstm(self, x_train):
        # Model architecture
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
        return model
