from itertools import combinations
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
from glupredkit.helpers.scikit_learn import process_data
from .base_model import BaseModel
import numpy as np
import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
import joblib
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
# Heterogeneous ensemble - linear models
# Stacked MLP and PLSR: 
# GBT
# SVR Linear

   

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.mlp_model = None
        self.plsr_model = None
        self.svm_model = None
        self.stacked_model = None


    def fit(self, X_train, Y_train):
        self.mlp_model = self._create_mlp()  
        self.plsr_model = self._create_plsr(X_train, Y_train)  
        self.svr_model = self._create_svr()
        estimators = [('mlp', self.mlp_model), ('plsr', self.plsr_model), ('svr', self.svr_model)]
        self.stacked_model = StackingRegressor(estimators=estimators, final_estimator=GradientBoostingRegressor())
        self.stacked_model.fit(X_train, Y_train)
        return self

    def predict(self, X_test):
        predictions = self.stacked_model.predict(X_test)
        return predictions 
    
    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    
    def _create_mlp(self):
        return MLPRegressor(hidden_layer_sizes=(40,40), max_iter=500, activation='relu', 
                            solver='adam', random_state=42, shuffle=False, 
                            early_stopping=True)
    
    def _create_plsr(self, X_train, Y_train):
        # maximum number of components based on the number of features
        componentmax = X_train.shape[1]
        component = np.arange(1, componentmax)
        rmse = []

        # loop over different numbers of components to find the best amount of components
        for i in component:
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
        
        # create a new PLS Regression model with the optimal number of components
        return PLSRegression(n_components)
    
    def _create_svr(self):
        pipeline = Pipeline([('regressor', SVR(tol=1, kernel='linear'))])
        # Define the parameter grid
        param_grid = {
            'regressor__C': [30],
            'regressor__epsilon': [0.015]
        }
        model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        return model
    '''
    def _create_gbt(self):
        pipeline = Pipeline([('regressor', xgb.XGBRegressor())])
        # Got better results with the following parameters but with significantly longer training time
        param_grid = {
            'regressor__n_estimators': [500, 1000, 2000],
            'regressor__max_depth': [6, 8, 10],
            'regressor__gamma': [0.01, 0.1, 0.2],
            'regressor__learning_rate': [0.001, 0.01, 0.05]
        }
        # Define GridSearchCV
        model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        return model
    '''