import numpy as np
import tensorflow as tf
import ast
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        # The recommended approach for saving and loading Keras models is to use Keras's built-in .save() and
        # Using legacy .h5 file type because .keras threw error with M1 Mac chip
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}.h5"

    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()

        sequences = np.array(sequences)
        targets = np.array(targets)

        print(sequences.shape)

        # Model architecture
        input_layer = Input(shape=(sequences.shape[1], sequences.shape[2]))
        lstm = LSTM(50, return_sequences=True)(input_layer)
        lstm = LSTM(50, return_sequences=True)(lstm)
        lstm = LSTM(50, return_sequences=False)(lstm)
        output_layer = Dense(1)(lstm)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            # optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9, clipnorm=1.0),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9),
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

            print("Train x", train_X.shape)
            print("Train y", train_Y.shape)
            print("val x", val_X.shape)
            print("val y", val_Y.shape)

            model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=20, batch_size=1,
                      callbacks=[early_stopping, reduce_lr])

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


    def process_data(self, df, num_lagged_features, numerical_features, categorical_features):
        return process_data(df, num_lagged_features, numerical_features, categorical_features)
