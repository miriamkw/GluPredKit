"""
This Temporal Convolutional Network (TCN) should be used together with data preprocessed using tf_keras.py.
"""
import numpy as np
import tensorflow as tf
import ast
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import TimeSeriesSplit
from tcn import TCN
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        # The recommended approach for saving and loading Keras models is to use Keras's built-in .save() and
        # Using legacy .h5 file type because .keras threw error with M1 Mac chip
        # Using current date in the file name to remove chances of equal file names
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/stl_ph-{prediction_horizon}_{safe_timestamp}.h5"

    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()

        sequences = np.array(sequences)
        targets = np.array(targets)

        dropout_conv = 0.1
        dropout_lstm = 0.2
        dropout_fc = 0.5
        seq_len = 24
        input_dim = 5
        out_dim = 1

        model = Sequential()

        # conv1 layer
        model.add(Convolution1D(8, 4, padding='causal', input_shape=(seq_len, input_dim), name='Shared_Conv_1'))  # 4
        model.add(MaxPooling1D(pool_size=2, name='Shared_MP_1'))

        # conv2 layer
        model.add(Convolution1D(16, 4, padding='causal', name='Shared_Conv_2'))
        model.add(MaxPooling1D(pool_size=2, name='Shared_MP_2'))
        model.add(Dropout(dropout_conv))

        # conv3 layer
        model.add(Convolution1D(32, 4, padding='causal', name='Shared_Conv_3'))
        model.add(MaxPooling1D(pool_size=2, name='Shared_MP_3'))
        model.add(Dropout(dropout_conv))

        # lstm layer
        model.add(LSTM(64, return_sequences=False, name='Shared_Layer_9'))  # , input_shape = (seq_len, input_dim)))
        model.add(Dropout(dropout_lstm))

        # fc1 layer
        model.add(Dense(256, name='dense_1'))  # 256
        model.add(Dropout(dropout_fc))

        # fc2 layer
        model.add(Dense(32, name='dense_2'))  # 32
        model.add(Dropout(dropout_fc))

        # fc3 layer
        model.add(Dense(out_dim, name="Output"))

        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00053)
        model.compile(optimizer=opt, loss='mean_absolute_error')

        # Split the data into 5 folds
        train_X = sequences
        train_Y = targets

        # Convert lists to numpy arrays if necessary
        #train_X = np.concatenate(train_X, axis=0)
        #train_Y = np.concatenate(train_Y, axis=0)

        # Fit the model with early stopping and reduce LR on plateau
        # model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=20,  batch_size=1)
        model.fit(train_X, train_Y, epochs=200, batch_size=128, shuffle=False, verbose=0, validation_split=0)

        model.save(self.model_path)

        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict(sequences)

        return [val[0] for val in predictions]

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
