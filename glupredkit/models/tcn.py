"""
This Temporal Convolutional Network (TCN) should be used together with data preprocessed using tf_keras.py.
"""
import numpy as np
import tensorflow as tf
import ast
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import TimeSeriesSplit
from tcn import TCN
from .base_model import BaseModel


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        # The recommended approach for saving and loading Keras models is to use Keras's built-in .save() and
        # Using legacy .h5 file type because .keras threw error with M1 Mac chip
        self.model_path = f"data/.keras_models/tcn_ph-{prediction_horizon}.h5"

    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Model architecture
        input_layer = Input(shape=(sequences.shape[1], sequences.shape[2]))

        # TCN layer
        tcn_out = TCN(nb_filters=50, return_sequences=False)(input_layer)

        output_layer = Dense(1)(tcn_out)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.999, beta_2=0.999, clipnorm=1.0),
            loss='mse')

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)

        # Split the data into 5 folds
        tscv = TimeSeriesSplit(n_splits=5)

        # Use first 4 folds for training and the last fold for validation
        for train_idx, val_idx in tscv.split(sequences):
            train_X, val_X = sequences[train_idx], sequences[val_idx]
            train_Y, val_Y = targets[train_idx], targets[val_idx]

            model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=20, batch_size=1,
                      callbacks=[early_stopping, reduce_lr])

        model.save(self.model_path)

        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam,
                                                                            "TCN": TCN})
        predictions = model.predict(sequences)

        return [val[0] for val in predictions]

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None
