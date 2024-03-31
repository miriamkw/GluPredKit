from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.models import Sequential
from datetime import datetime
import tensorflow as tf
import numpy as np
import keras
import ast

# Arcitecture referenced from paper "Deep Physiological Model for Blood Glucose Prediction in T1DM Patients"
# doi: 10.3390/s20143896
# PH30 - 23.825358
class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/physio3_ph-{prediction_horizon}_{safe_timestamp}.h5"
        self.model = None

    def fit(self, x_train, y_train):
        x_train = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        y_train = y_train.tolist()
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print("x_train[0] shape: ", x_train.shape) # x_train shape:  (xxxxx, 36, 4)

        # Define input shape and LSTM parameters
        time_span = 12  # 1 hours (however the paper used 9 hours of time span)
        mem_cells = 100

        # Define model
        input_models = []
        lstm_outputs = []
        for _ in range(4):
            input_layer = Input(shape=(time_span, 1))
            input_models.append(input_layer)
            lstm_output = self._create_model(mem_cells, x_train.shape)(input_layer)
            lstm_outputs.append(lstm_output)

        # Concatenate output layers
        concatenated = Concatenate(axis=-1)(lstm_outputs[1:])
        added2 = Concatenate(axis=-1)([lstm_outputs[0], concatenated])
        dense_layer = Dense(1)(added2)
        model = keras.Model(inputs=input_models, outputs=dense_layer)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        
        # Define callbacks
        early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min')
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss', mode='min')

        # Training the model
        model.fit([x_train[:, :, 0:1], x_train[:, :, 1:2], x_train[:, :, 2:3], x_train[:, :, 3:4]],
                    y_train,
                    epochs=100,
                    batch_size=1,
                    validation_split=0.3,
                    verbose=1,
                    callbacks=[early_stopping, reduce_lr],
                    shuffle=False)
            
        model.save(self.model_path)
        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        sequences = np.array(sequences)

        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict([sequences[:, :, 0:1], sequences[:, :, 1:2], sequences[:, :, 2:3], sequences[:, :, 3:4]])
        return [val[0] for val in predictions]


    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    
    def _create_model(self, mem_cells, input_shape):
        model = Sequential()
        model.add(LSTM(mem_cells, activation='relu', input_shape=(input_shape[1], 1)))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(3))
        return model
