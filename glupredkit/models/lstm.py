import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.models import Sequential
# from sklearn.model_selection import TimeSeriesSplit
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
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"

    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Model architecture
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(sequences.shape[1], sequences.shape[2])))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
        # Convert lists to numpy arrays if necessary
        train_X = sequences
        train_Y = targets

        # Fit the model with early stopping and reduce LR on plateau
        model.fit(train_X, train_Y, epochs=100)
        model.save(self.model_path)

        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict(sequences)

        return [val[0] for val in predictions]

    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
