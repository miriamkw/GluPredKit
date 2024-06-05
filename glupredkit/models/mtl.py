"""
Multitask learning (CRNN), from: https://ceur-ws.org/Vol-2675/paper19.pdf
GitHub: https://github.com/jsmdaniels/ecai-bglp-challenge
"""
import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
from keras.models import Model as KerasModel
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Dropout
from keras.layers import LSTM
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
        self.model_path = f"data/.keras_models/mtl_ph-{prediction_horizon}_{safe_timestamp}.h5"

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
        main_input = Input(shape=input_shape, dtype='float32', name='Input')

        # conv1 layer
        x = Convolution1D(8, 4, padding='causal', name='Conv_1')(main_input)
        x = MaxPooling1D(pool_size=2, name='MP_1')(x)

        # conv2 layer
        x = Convolution1D(16, 4, padding='causal', name='Conv_2')(x)
        x = MaxPooling1D(pool_size=2, name='MP_2')(x)
        x = Dropout(dropout_conv, name='Dropout_1')(x)

        # conv3 layer
        x = Convolution1D(32, 4, padding='causal', name='Conv_3')(x)
        x = MaxPooling1D(pool_size=2, name='MP_3')(x)
        x = Dropout(dropout_conv, name='Dropout_2')(x)

        # lstm layer
        x = LSTM(64, return_sequences=False, name='LSTM_Layer')(x)
        x = Dropout(dropout_lstm, name='Dropout_3')(x)

        # Fully connected layers
        x = Dense(256, name='Dense_1')(x)
        x = Dropout(dropout_fc, name='Dropout_4')(x)
        x = Dense(32, name='Dense_2')(x)
        x = Dropout(dropout_fc, name='Dropout_5')(x)

        # Output layer
        output = Dense(out_dim, name='Output')(x)

        model = KerasModel(inputs=main_input, outputs=output)
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00053)
        model.compile(optimizer=opt, loss='mean_absolute_error')

        # Split the data into 5 folds
        train_X = sequences
        train_Y = targets

        # Fit the model with early stopping and reduce LR on plateau
        model.fit(train_X, train_Y, epochs=epochs, batch_size=128, shuffle=False, verbose=0, validation_split=0)

        model.save(self.model_path)
        self.is_fitted = True
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
