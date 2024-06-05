"""
Double lstm, from: https://ieeexplore.ieee.org/document/8856940
Open code: http://smarthealth.cs.ohio.edu/nih.html
"""
import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
from tensorflow.keras.layers import LSTM, Dense, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.second_layer_size = 20
        self.ph = prediction_horizon
        self.mode = 'basic'
        self.num_layers = 2

        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        safe_timestamp = safe_timestamp.replace('.', '_')
        self.model_path = f"data/.keras_models/double_lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"

    def build_model(self, x1_shape, x2_shape, n_outputs):
        # create feedforward module
        input1 = Input(shape=(x1_shape[1], x1_shape[2]), name='input_with_bgl')
        input2 = Input(shape=(x2_shape[1], x2_shape[2]), name='input_without_bgl')

        last_layer = input1
        for i in range(self.num_layers):
            _output_seq = i < (self.num_layers - 1)
            last_layer, last_h, last_c = LSTM(units=20,
                                              activation='tanh',
                                              recurrent_activation='sigmoid',
                                              implementation=2,
                                              kernel_initializer='glorot_uniform',
                                              dropout=0.0,
                                              return_state=True,
                                              return_sequences=_output_seq)(last_layer)
        # create attention module
        mapped_h = Dense(units=20, activation='tanh')(last_h)
        mapped_c = Dense(units=20, activation='tanh')(last_c)

        layer2 = LSTM(units=self.second_layer_size,
                      activation='tanh',
                      recurrent_activation='sigmoid',
                      implementation=2,
                      kernel_initializer='glorot_uniform',
                      dropout=0.0)(input2, initial_state=[mapped_h, mapped_c])

        _feed_to_last_layer = layer2
        outputs = [Dense(n_outputs, activation='linear')(_feed_to_last_layer)]

        model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)
        return model

    def _fit_model(self, x_train, y_train, epochs=10, *args):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_train['target']]

        sequences = np.array(sequences)
        targets = np.array(targets)

        x_train1 = sequences[:, :, 0:1]  # BGL data, keep the third dimension by using 0:1
        x_train2 = sequences[:, :, 1:]  # all data without BGL data

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model = self.build_model(x_train1.shape, x_train2.shape, targets.shape[1])
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                                                clipnorm=1.0),
                      loss='mean_squared_error', metrics=['mean_absolute_error'])
        model.fit([x_train1, x_train2], targets, epochs=epochs, callbacks=[early_stopping],
                  validation_split=0.2)

        model.save(self.model_path)
        return self

    def _predict_model(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        x_test1 = sequences[:, :, 0:1]
        x_test2 = sequences[:, :, 1:]
        x_test = [x_test1, x_test2]

        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict(x_test)

        return predictions

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
