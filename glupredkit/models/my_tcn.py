from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tcn import TCN, tcn_full_summary
import ast
import numpy as np
import tensorflow as tf
from datetime import datetime
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data
from sklearn.model_selection import TimeSeriesSplit

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/tcn_ph-{prediction_horizon}_{safe_timestamp}.h5"

    def _build_model(self, input_shape):
        model = Sequential()
        input = Input(shape=(input_shape[1], input_shape[2]))
        # nb_filters: the more the better as long as do not overfit
        # kernel_size: ideally small 2 or 3 (short term dependencies)
        # dilations: should cover time steps and should use the number that is power of 2
        output = (TCN(nb_filters=64, kernel_size=3, dilations=[8, 16, 32], padding='causal')(input))
        output = Dense(1)(output)
        model = tf.keras.Model(inputs=input, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
        tcn_full_summary(model, expand_residual_blocks=False)
        return model

    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()
        x_train = np.array(sequences)
        y_train = np.array(targets)
        model = self._build_model(x_train.shape)

        tscv = TimeSeriesSplit(n_splits=5)
        train_X, train_Y = [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(x_train)):
            if fold < 4:
                train_X.append(x_train[train_idx])
                train_Y.append(y_train[train_idx])
            else:
                val_X, val_Y = x_train[val_idx], y_train[val_idx]

        train_X = np.concatenate(train_X, axis=0)
        train_Y = np.concatenate(train_Y, axis=0)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)

        model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=50, batch_size=1, callbacks=[early_stopping, reduce_lr])
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

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    

'''
kernel size    3 | with folding
30ph -        23 |
60ph -        41 |
'''