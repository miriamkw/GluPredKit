"""
Bidirectional LSTM, from: https://ceur-ws.org/Vol-2675/paper12.pdf
GitHub: https://github.com/meneghet/BGLP_challenge_2020
"""
from datetime import datetime
from .base_model import BaseModel
import tensorflow as tf
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.regularizers import l2
from keras.models import Sequential
from glupredkit.helpers.tf_keras import process_data
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import ast


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.lookback = 15
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"

    def fit(self, x_train, y_train, epochs=20):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_train['target']]

        x_train = np.array(sequences)
        y_train = np.array(targets)

        model = self._build_model(x_train.shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss', mode='min')
        model.fit(x_train, y_train, epochs=epochs, shuffle=False, verbose=True, callbacks=[early_stopping, reduce_lr],
                  validation_split=0.2)
        model.save(self.model_path)
        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        x_test = np.array(sequences)
        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict(x_test)
        return [val[0] for val in predictions]

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def _build_model(self, input_shape, lr=0.001, verbose=True):
        dropout = 0.1
        recurrent_dropout = 0.2

        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout,
                                     kernel_regularizer=l2(0.001)), input_shape=(input_shape[1], input_shape[2])))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        if verbose:
            model.summary()

        return model

    def _split_sequences(self, X_data, Y_data, n_outputs=1):
        # Adjust input and output data
        sequences = np.concatenate((X_data, np.array(Y_data).reshape(-1, 1)), axis=1)

        # Prepare data for LSTM
        X, y = list(), list()

        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.lookback
            # check if we are beyond the dataset
            if (end_ix + n_outputs - 1) > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[(end_ix - 1):(end_ix - 1 + n_outputs), -1]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)

