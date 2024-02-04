from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
# from glupredkit.helpers.scikit_learn import process_data
from glupredkit.helpers.tf_keras import process_data
from .base_model import BaseModel
import ast
from keras.models import Sequential
import numpy as np
from keras.layers import Dense

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.ph = prediction_horizon // 5 # 5 minute intervals, distance_target
        self.epochs = 100
        self.model = None

    def fit(self, x_train, y_train):
        x_train = x_train['sequence'].apply(ast.literal_eval) 
        x_train = np.array(x_train.tolist())
        x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten each sample
        y_train = y_train.tolist()
        self.model = MLPRegressor(hidden_layer_sizes=(40,40), max_iter=500, activation='relu', 
                                  solver='adam', random_state=42, shuffle=False, 
                                  early_stopping=True)
        
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        # print("x_test: ", x_test)
        x_test = x_test['sequence'].apply(ast.literal_eval) 
        x_test = np.array(x_test.tolist())
        x_test = x_test.reshape(x_test.shape[0], -1)  # Flatten each sample
        predictions = self.model.predict(x_test)
        return predictions 
    
    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)