import numpy as np
import tensorflow as tf
import ast
from tensorflow.keras.layers import LSTM, Dense, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.second_layer_size = 20
        self.ph = prediction_horizon
        self.mode = 'basic'
        self.double_output = False
        self.double_lstm_model = None
        self.num_layers = 2

    def build_model(self, input_shape):
        # create feedforward module
        print("input_shape: ", input_shape)
        input1 = Input(shape=(input_shape[1], input_shape[2]), name='input_with_bgl')
        input2 = Input(shape=(input_shape[1], 5), name='input_without_bgl')

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

        if self.mode == 'basic':
            _feed_to_last_layer = layer2
        elif self.mode == 'feed_last_h_to_predictor':
            _feed_to_last_layer = concatenate([mapped_h, layer2])
        else:
            raise ValueError("The provided mode is not recognized: ", self.mode)

        outputs = [Dense(1, activation='linear')(_feed_to_last_layer)]

        if self.double_output:
            outputs.append(Dense(1, activation='linear')(last_h))

        self.double_lstm_model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)
        return self.double_lstm_model

    def fit(self, x_train, y_train):
        # convert string representationof list of lists back into an actual list of lists.
        x_train = x_train['sequence'].apply(ast.literal_eval) 
        x_train = np.array(x_train.tolist())
        
        x_train1 = x_train[:, :, 0:1]  # BGL data, keep the third dimension by using 0:1 instead of just 0
        x_train2 = x_train[:, :, 1:]  # all data without BGL data

        if self.double_output:
            y_train = [y_train, y_train]

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        self.build_model(x_train1.shape)
        self.double_lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        self.double_lstm_model.fit([x_train1, x_train2], y_train, epochs=350,  callbacks=[early_stopping], validation_split=0.2)
        return self

    def predict(self, x_test):
        x_test = x_test['sequence'].apply(ast.literal_eval) 
        x_test = np.array(x_test.tolist())
        
        x_test1 = x_test[:, :, 0:1]
        x_test2 = x_test[:, :, 1:]

        x_test = [x_test1, x_test2]
        return self.double_lstm_model.predict(x_test, batch_size=32)


    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    