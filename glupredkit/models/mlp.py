import itertools
import time
from sklearn.base import BaseEstimator
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
from glupredkit.helpers.scikit_learn import process_data
from .base_model import BaseModel

from keras.models import Sequential
from sys import stdout
import numpy as np
from keras.layers import Dense

# MLPs are commonly used for tasks where the order or sequence of data is not crucial, 
# such as image classification, speech recognition, or basic regression problems
# Assumption: MLP will generate low performance
class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.ph = prediction_horizon // 5 # 5 minute intervals, distance_target
        self.epochs = 100
        self.model = None

    def fit(self, X_train, Y_train):
        self.model = Sequential()
        self.model.add(Dense(100, activation='relu', input_dim=X_train.shape[1])) # input layer
        self.model.add(Dense(1)) # output layer, Y_train.shape[0]
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, Y_train,  epochs=self.epochs, verbose=1)
        return self

    def predict(self, X_test):
        predictions = self.model.predict(X_test, verbose=1)
        return predictions 
    
    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)