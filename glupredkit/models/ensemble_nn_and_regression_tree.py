from sklearn.base import BaseEstimator
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
from glupredkit.helpers.scikit_learn import process_data
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from keras.models import Sequential
from keras.layers import Dense
from skrebate import ReliefF
from hyperopt import hp, fmin, tpe, Trials
from scipy.interpolate import interp1d
from .base_model import BaseModel

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.ph = prediction_horizon
        self.window = 16
        self.num_iteration = 10 # as per the source code, may need adjustment
        self.snn_model = None
        self.eim_model = RandomForestRegressor(n_estimators=100)

    def fit(self, X_train, Y_train):
        self.snn_model = self._create_shallow_nn(X_train.shape[1:])
        # for _ in range(self.num_iteration):
        # X = self._prep_nn(X_train, self.window, self.ph)
        self.snn_model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        # Train corrective model for each iteration
        cgm_hat_train = self.snn_model.predict(X_train)
        cgm_array = Y_train.values
        # max_features = 35
        error = cgm_array - cgm_hat_train.flatten()
        self._train_error_imputation_module(X_train, error)

        return self

    def predict(self, X_test):
        snn_predictions = self.snn_model.predict(X_test)
        eim_predictions = self.eim_model.predict(X_test)
        final_predictions = self._combine_predictions(snn_predictions, eim_predictions)
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
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Output layer for predicting BG values
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Train the shallow neural network
    def _train_shallow_nn(self, X, Y):
        model = self._create_shallow_nn(X.shape[1:])
        # Train the model
        model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2, verbose=1) # Example hyperparameters
        return model

    def _train_error_imputation_module(self, features, error):
        # max_features = 35
        # error = features_array - target.flatten()  # Calculating error from features and target
        
        # feature ranking using ReliefF
        if self.ph == 6:  # 30 min prediction horizon
            feature_selector = ReliefF(n_neighbors=10)  # as per the source code, may adjust n_neighbors if needed
        else:
            feature_selector = ReliefF(n_neighbors=30)

        # print("values ", features.values)
        # print("reshaped ", features.values.reshape(-1, 1))
        feature_weights = feature_selector.fit(features.values, error).feature_importances_
        ranked_indices = np.argsort(feature_weights)[::-1]  # Reverse sort indices
        
        print("ranked_indices: ", ranked_indices)
        # best_feature_indices = ranked_indices[:max_features]
        best_feature_indices = ranked_indices[:]
        
        selected_features = features.values[:, best_feature_indices]

        # model training (using RandomForestRegressor)
        rf_regressor = RandomForestRegressor(n_estimators=100)
        self.eim_model.fit(selected_features, error)
        
        return self.eim_model

    def _combine_predictions(self, snn_predictions, eim_predictions):
        return snn_predictions + eim_predictions

    '''
    takes a sequential data stream, breaks it down into smaller segments of a defined window size, 
    arranges segments into a structured format for neural networks.
    '''
    def _prep_nn(self, signal, ord_val, ph_val):
        X = np.zeros((len(signal) - ord_val - ph_val + 1, ord_val))
        for k in range(ord_val, len(signal) - ph_val + 1):
            X[k - ord_val] = signal[k - ord_val:k]
        return X