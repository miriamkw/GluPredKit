from sklearn.linear_model import HuberRegressor, RidgeCV, LassoLarsIC, LinearRegression
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

# Homogeneous ensemble - linear models
# RidgeCV: Classical Linear Model with built-in cross-validation
# HuberRegressor: Outlier-robust
# LassoLarsIC: Regressor with varianble selection

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.ridge_model = None
        self.huber_model = None
        self.lassolars_model = None
        self.stacked_model = None

    def fit(self, x_train, y_train):
        self.ridge_model = self._create_ridge(x_train, y_train)
        self.huber_model = self._create_huber(x_train, y_train)
        self.lassolars_model = self._create_lassolars(x_train, y_train)
        base_models = [
            ('ridge', self.ridge_model),
            ('huber', self.huber_model),
            ('lasso', self.lassolars_model)
        ]
        meta_model = LinearRegression()
        self.stacked_model = VotingRegressor(
            estimators=base_models,
            #final_estimator=meta_model,
            #cv=5
        )
        self.stacked_model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        y_pred = self.stacked_model.predict(x_test)
        return y_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    
    def _create_ridge(self, x_train, y_train):
        '''
        pipeline = Pipeline([
            ('regressor', Ridge())
        ])
        param_grid = {
            'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'regressor__tol': [1e-4, 1e-3, 1e-2]
        }
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(x_train, y_train)
        return grid_search.best_estimator_
        '''
        ridgecv = RidgeCV()
        # ridgecv.fit(x_train, y_train)
        return ridgecv
    
    def _create_huber(self, x_train, y_train):
        pipeline = Pipeline([
            ('regressor', HuberRegressor())
        ])
        param_grid = {
            'regressor__epsilon': [1.0, 1.2, 1.4, 1.6],
            'regressor__alpha': [0.00001, 0.0001, 0.0005],
            'regressor__max_iter': [2000] 
        }
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(x_train, y_train)
        return grid_search.best_estimator_
    
    def _create_lassolars(self, x_train, y_train):
        model = LassoLarsIC() #.fit(x_train, y_train)
        return model