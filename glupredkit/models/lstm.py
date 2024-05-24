import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten, concatenate, Input, Masking, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import TimeSeriesSplit
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
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"
        self.input_shape = None
        self.num_outputs = None

    def _fit_model(self, x_train, y_train, epochs=20, *args):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_train['target']]

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Determine the number of outputs
        self.input_shape = (sequences.shape[1], sequences.shape[2])
        self.num_outputs = targets.shape[1]  # Assuming targets is 2D: [samples, outputs]

        # Model architecture
        input_layer = Input(shape=(sequences.shape[1], sequences.shape[2]))
        lstm = LSTM(50, return_sequences=True)(input_layer)
        lstm = LSTM(50, return_sequences=True)(lstm)
        lstm = LSTM(50, return_sequences=False)(lstm)
        output_layer = Dense(self.num_outputs)(lstm)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0),
            loss='mse')

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)

        # Split the data into 5 folds
        tscv = TimeSeriesSplit(n_splits=5)

        train_X, train_Y = [], []
        # Use first 4 folds for training and the last fold for validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(sequences)):
            if fold < 4:  # Accumulate the first 4 folds for training
                train_X.append(sequences[train_idx])
                train_Y.append(targets[train_idx])
            else:  # Use the 5th fold for validation
                val_X, val_Y = sequences[val_idx], targets[val_idx]

        # Convert lists to numpy arrays if necessary
        train_X = np.concatenate(train_X, axis=0)
        train_Y = np.concatenate(train_Y, axis=0)

        # Fit the model with early stopping and reduce LR on plateau
        model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=epochs, batch_size=1,
                  callbacks=[early_stopping, reduce_lr])

        model.save(self.model_path)
        return self

    def _predict_model(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict(sequences)
        predictions = predictions.tolist()

        return predictions

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

