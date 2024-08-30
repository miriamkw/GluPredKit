"""
Single-task learning (CRNN), from: https://ceur-ws.org/Vol-2675/paper19.pdf
GitHub: https://github.com/jsmdaniels/ecai-bglp-challenge
"""
import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, LSTM, Dense, Dropout
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
        safe_timestamp = safe_timestamp.replace('.', '_')
        self.model_path = f"data/.keras_models/stl_ph-{prediction_horizon}_{safe_timestamp}.h5"

    def _fit_model(self, x_train, y_train, epochs=20, *args):
        # TODO: Implement transfer learning on population to personalization
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_train['target']]

        sequences = np.array(sequences)
        targets = np.array(targets)

        dropout_conv = 0.1
        dropout_lstm = 0.2
        dropout_fc = 0.5

        out_dim = targets.shape[1]
        input_shape = (sequences.shape[1], sequences.shape[2])

        model = Sequential()

        # conv1 layer
        model.add(Convolution1D(8, 4, padding='causal', input_shape=input_shape, name='Shared_Conv_1'))  # 4
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
        model.add(LSTM(64, return_sequences=False, name='Shared_Layer_9'))
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

        # Fit the model with early stopping and reduce LR on plateau
        model.fit(train_X, train_Y, epochs=epochs, batch_size=128, shuffle=False, verbose=0, validation_split=0)

        model.save(self.model_path)
        return self

    def _predict_model(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict(sequences)

        return predictions

    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
