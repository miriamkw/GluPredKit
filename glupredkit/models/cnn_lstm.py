import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.models import Sequential
# from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data
#from glupredkit.helpers.scikit_learn import process_data

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        # The recommended approach for saving and loading Keras models is to use Keras's built-in .save() and
        # Using legacy .h5 file type because .keras threw error with M1 Mac chip
        # Using current date in the file name to remove chances of equal file names
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"
        self.model = None
        
    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()

        sequences = np.array(sequences)
        targets = np.array(targets)
   
        model = self.__create_model(sequences.shape)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
        # Convert lists to numpy arrays if necessary
        train_X = sequences
        train_Y = targets

        # Define callbacks
        early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min')
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss', mode='min')

        # Fit the model with early stopping and reduce LR on plateau
        model.fit(train_X, train_Y, epochs=100, callbacks=[early_stopping, reduce_lr])
        # self.model = model
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
    
    
    # Define CNN-LSTM model
    def __create_model(self, input_shape, filters=64, kernel_size=3, pool_size=2, lstm_units=64, dropout=0.2):
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(input_shape[1], input_shape[2])))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(units=lstm_units, dropout=dropout))
        model.add(Dense(1))  # Output layer with single output for glucose prediction
        return model