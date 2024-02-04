import ast
from sklearn.base import BaseEstimator
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
# from glupredkit.helpers.scikit_learn import process_data
from glupredkit.helpers.tf_keras import process_data
from .base_model import BaseModel


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sys import stdout
import numpy as np
from sklearn.cross_decomposition import PLSRegression


# MLPs are commonly used for tasks where the order or sequence of data is not crucial, 
# such as image classification, speech recognition, or basic regression problems
# Assumption MLP will generate low performance
class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.ph = prediction_horizon // 5 # 5 minute intervals, distance_target
        self.epochs = 100
        self.model = None

    def fit(self, X_train, Y_train):
        x_train = X_train['sequence'].apply(ast.literal_eval) 
        x_train = np.array(x_train.tolist())
        X_train = x_train.reshape(x_train.shape[0], -1)  # Flatten each sample
        Y_train = np.array(Y_train.tolist())
        # maximum number of components based on the number of features
        componentmax = X_train.shape[1]
        component = np.arange(1, componentmax)
        rmse = []

        # loop over different numbers of components to find the best amount of components
        for i in component:
            # Create a PLS Regression model with i components
            pls = PLSRegression(n_components=i)
            pls.fit(X_train, Y_train)
            Y_pred_train = pls.predict(X_train)
            msecv = mean_squared_error(Y_train, Y_pred_train)
            rmsecv = np.sqrt(msecv)
            rmse.append(rmsecv)

        # find the number of components that minimizes RMSE
        msemin = np.argmin(rmse)
        # set the optimal number of components
        n_components = msemin + 1
        
        # Create a new PLS Regression model with the optimal number of components
        self.model = PLSRegression(n_components)
        self.model.fit(X_train, Y_train)
        return self

    def predict(self, X_test):
        x_test = X_test['sequence'].apply(ast.literal_eval) 
        x_test = np.array(x_test.tolist())
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        predictions = self.model.predict(x_test_flat)
        return predictions
    
    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)