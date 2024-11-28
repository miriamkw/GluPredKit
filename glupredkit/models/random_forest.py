from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.subject_ids = None
        self.models = []  # One model per subject
        self.best_params = []  # Store best params per subject

    def _fit_model(self, x_train, y_train, *args):
        self.subject_ids = x_train['id'].unique()
        
        # Quick but meaningful parameter grid
        param_grid = {
            'estimator__n_estimators': [50, 100],  # Number of trees
            'estimator__max_depth': [5, 10],       # Tree depth
            'estimator__min_samples_split': [5, 10] # Min samples to split node
        }
        
        for subject_id in self.subject_ids:
            print(f"Training for subject {subject_id}")
            x_train_subject = x_train[x_train['id'] == subject_id]
            y_train_subject = y_train[x_train['id'] == subject_id]
            
            # Base random forest with reasonable defaults
            rf = RandomForestRegressor(
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            
            # Wrap with MultiOutputRegressor
            multi_rf = MultiOutputRegressor(rf)
            
            # GridSearch with 3-fold CV (faster than 5-fold)
            grid_search = GridSearchCV(
                multi_rf,
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1  # Parallel processing
            )
            
            # Fit and store the best model
            grid_search.fit(x_train_subject, y_train_subject)
            self.models.append(grid_search.best_estimator_)
            self.best_params.append(grid_search.best_params_)
            
            print(f"Best parameters for subject {subject_id}: {grid_search.best_params_}")
            
        return self

    def _predict_model(self, x_test):
        y_pred = []
        ids_list = x_test.id.unique()

        for curr_id in ids_list:
            model_index = np.where(self.subject_ids == curr_id)[0][0]
            subset_df = x_test[x_test['id'] == curr_id]
            predictions = self.models[model_index].predict(subset_df)
            y_pred += predictions.tolist()
            
        return np.array(y_pred)

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def get_best_params(self):
        """Return dictionary of best parameters per subject"""
        #return dict(zip(self.subject_ids, self.best_params))
        return 

