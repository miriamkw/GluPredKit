import numpy as np
import tensorflow as tf
import ast
from datetime import datetime

from keras.src.layers import Flatten
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
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
        self.input_shape = None
        self.num_outputs = None

    def fit(self, x_train, y_train):

        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_train['target']]

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Determine the number of outputs
        self.input_shape = (sequences.shape[1], sequences.shape[2])
        self.num_outputs = targets.shape[1]  # Assuming targets is 2D: [samples, outputs]

        model = self.build_model(self.input_shape, self.num_outputs)

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                      loss=self.loss_with_constraints(model))

        model.fit(sequences, targets, epochs=50)
        model.save(self.model_path)

        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        model = self.load_model()
        predictions = model.predict(sequences)
        predictions = predictions.tolist()

        return predictions

    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def build_model(self, input_shape, num_outputs, l2_reg=0.001, dropout_rate=0.3):
        model = Sequential([
            Masking(mask_value=-1., input_shape=input_shape),
            # LSTM(64, input_shape=input_shape, return_sequences=False, kernel_regularizer=l2(l2_reg)),
            LSTM(200, activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_reg)),
            Dense(100, activation='relu'),

            Dropout(dropout_rate),
            Dense(num_outputs, kernel_regularizer=l2(l2_reg))
        ])
        return model

    def loss_with_constraints(self, model, lambda_penalty=0.4):
        def loss(y_true, y_pred):
            # Calculate the base loss (mean squared error in this example)

            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            """
            # Access the weights of the specified layer
            # layer.weights[0] accesses the weight matrix; layer.weights[1] would access biases
            weights = model.layers[-1].weights[0]

            # Calculate penalties
            # Penalty for the second feature (carbs) weights if they are negative
            penalty_2nd_feature = tf.reduce_sum(tf.maximum(0., -weights[:, 1]))

            # Penalty for the third feature (insulin) weights if they are positive
            penalty_3rd_feature = tf.reduce_sum(tf.maximum(0., weights[:, 2]))

            # Total penalty
            total_penalty = lambda_penalty * (penalty_2nd_feature + penalty_3rd_feature)
            """
            # Calculate the base loss (mean squared error in this example)
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # Calculate the absolute errors
            errors = K.abs(y_true - y_pred)

            # Compute weights based on the target values
            # Weights increase as targets move away from the mean
            weights = K.abs(y_true - K.mean(y_true))

            # Apply the weights to the errors
            weighted_errors = weights * errors

            # Calculate the mean of the weighted errors
            weighted_error_loss = K.mean(weighted_errors)

            # Combine the MSE loss with the weighted error loss
            # Adjust the balance with a lambda_penalty parameter
            total_loss = mse_loss + lambda_penalty * weighted_error_loss

            return total_loss

            # total_penalty = 0

            # Total loss is the sum of MSE loss and the penalty
            # return mse_loss + total_penalty

        return loss

    def load_model(self):
        model = self.build_model(self.input_shape, self.num_outputs)

        loaded_model = tf.keras.models.load_model(self.model_path, custom_objects={
            "Adam": tf.keras.optimizers.legacy.Adam,
            "loss": self.loss_with_constraints(model)
        })
        return loaded_model

