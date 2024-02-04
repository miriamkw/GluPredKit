from sklearn.base import BaseEstimator
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
#from glupredkit.helpers.scikit_learn import process_data
from glupredkit.helpers.tf_keras import process_data
import ast
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from keras.models import Sequential
from keras.layers import Dense, LSTM
from skrebate import ReliefF
from hyperopt import hp, fmin, tpe, Trials
from scipy.interpolate import interp1d
from .base_model import BaseModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit

# def create_model(optimizer, input_shape):
#     model = Sequential()
#     model.add(Dense(12, input_dim=input_shape, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer='Nadam')
#     return model
class Model(BaseModel):
    def __init__(self, prediction_horizon):
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"

        self.snn_model = None
        self.eim_model = RandomForestRegressor(n_estimators=30)
        self.best_indices = None

    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()

        sequences = np.array(sequences)
        targets = np.array(targets)

        self.snn_model = self._create_shallow_nn(sequences.shape)
        

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
        self.snn_model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=10, batch_size=1,
                  callbacks=[early_stopping, reduce_lr])
        
        snn_pred = self.snn_model.predict(train_X)
        
        print("snn_pred: ", snn_pred)
        print("snn_pred.shape: ", snn_pred.shape)
        train_Y = np.reshape(train_Y, (train_Y.shape[0], 1))
        last_timestep_snn_pred = snn_pred[:, -1, :]
        # calculate error
        error = train_Y - last_timestep_snn_pred
        self._train_error_imputation_module(last_timestep_snn_pred, error)
        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        x_test = np.array(sequences)

        snn_predictions = self.snn_model.predict(x_test)
        last_timestep_snn_pred = snn_predictions[:, -1, :]

        last_timestep_x_test = x_test[:, -1, :]
        eim_predictions = self.eim_model.predict(last_timestep_snn_pred)
  
        final_predictions = self._combine_predictions(last_timestep_snn_pred, eim_predictions)
        return final_predictions

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    
#----------------------------------------------------------------
# private methods
#----------------------------------------------------------------
    # shallow neural network (SNN)
    def _create_shallow_nn(self, input_shape):
        model = Sequential()
        # Add layers to the neural network
        model.add(Dense(16, activation='relu', input_shape=(input_shape[1], input_shape[2])))
        model.add(Dense(1, activation='linear'))
        # Compile the model
        model.compile(optimizer='Nadam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model

    def _train_error_imputation_module(self, features, error):
        max_features = 4
        last_timestep_features = features[:, -1]

        # # features.columns = ['CGM', 'carbs', 'bolus', 'basal', 'gsr', 'skin_temp']  
        # feature_selector = ReliefF(n_features_to_select=4, n_neighbors=10) # 30
        # feature_weights = feature_selector.fit(features, error).feature_importances_
        # ranked_indices = np.argsort(feature_weights)[::-1]  # Reverse sort indices
        # self.best_indices = ranked_indices[:max_features]
        # selected_features = features[:, self.best_indices]
        # print("selected_features:", selected_features.shape)

        self.eim_model.verbose = 1 
        self.eim_model.fit(features, error) 
        
        return self.eim_model

    def _combine_predictions(self, snn_predictions, eim_predictions):
        eim_predictions = np.array(eim_predictions).reshape(-1, 1)
        print("eim_predictions:", eim_predictions)
        return snn_predictions + eim_predictions

    '''
    takes a sequential data stream, breaks it down into smaller segments of a defined window size, 
    arranges segments into a structured format for neural networks.
    '''
    def _prep_nn(self, data_stream, window):
        X = np.zeros((len(data_stream), window))
        for i in range(0, window):
            X[i] = ([data_stream.iloc[i]] * i) + ([data_stream.iloc[i]] * (window - i))
        for k in range(window, len(data_stream)):
            X[k] = data_stream[k - window:k]
        return X
 