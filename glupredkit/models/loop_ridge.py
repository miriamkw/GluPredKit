from glupredkit.models.base_model import BaseModel
from glupredkit.models.loop_algorithm import Model as LoopModel
from glupredkit.helpers.scikit_learn import process_data
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import ctypes
import pandas as pd
import numpy as np
import json

class Model(BaseModel):

    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.basal = None
        self.insulin_sensitivity_factor = None
        self.carb_ratio = None
        self.loop_model = None
        self.ridge_model = None
        self.ridge_features = None
        self.transformed_features = None
        self.standard_scaler = None


    def _fit_model(self, x_train, y_train, *args):
        # TODO: Train per id, change therapy settings to lists
        self.loop_model = LoopModel(self.prediction_horizon).fit(x_train, y_train)
        self.loop_model.basal = 0.75
        self.loop_model.insulin_sensitivity_factor = 66.6
        self.loop_model.carb_ratio = 9

        # TODO: Remove this later
        n_samples = 8000
        x_train = x_train.iloc[:n_samples, :]
        y_train = y_train.iloc[:n_samples, :]

        # Get predictions after 15 minutes, this shall be the target value for the error prediction
        predictions = self.loop_model.predict(x_train)
        predictions_15 = [val[2] for val in predictions]

        # Train the ridge model on the predicted errors
        self.transformed_features = ['CGM', 'active_carbs', 'active_insulin']
        self.ridge_features = ['active_carbs', 'active_insulin']  # TODO: Maybe add what if features in the 15 min
        self.ridge_features += [val for val in x_train.columns if val.startswith('hour') and 'what_if' not in val]
        self.ridge_features += ['interaction_CGM_carbs', 'interaction_CGM_insulin']

        # Train a standard scaler
        self.standard_scaler = StandardScaler()
        X = x_train[self.transformed_features].values
        self.standard_scaler.fit(X)

        ridge_df = self.add_interaction_terms(x_train)
        ridge_df['y_true_15'] = y_train['target_15']
        ridge_df['y_pred_15'] = predictions_15
        ridge_df['target'] = ridge_df['y_pred_15'] - ridge_df['y_true_15']
        ridge_df.drop(columns=['y_true_15', 'y_pred_15'], inplace=True)

        # Train a ridge regressor
        # TODO: Add constraints to the coefficients
        # TODO: Change the parameter grid scoring method?
        # Define the parameter grid
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
        self.ridge_model = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=3)

        self.ridge_model.fit(ridge_df[self.ridge_features], ridge_df['target'])

        # Print coefficients with feature names
        coefficients = self.ridge_model.best_estimator_.coef_
        intercept = self.ridge_model.best_estimator_.intercept_

        print("Intercept:", intercept)
        for feature_name, coef in zip(self.ridge_features, coefficients):
            print(f"{feature_name}: {coef}")

        return self

    def _predict_model(self, x_test):
        loop_predictions = self.loop_model.predict(x_test)

        df = self.add_interaction_terms(x_test)
        ridge_predictions = self.ridge_model.best_estimator_.predict(df[self.ridge_features])
        combined_predictions = []

        # Here we add the error to the three first predictions
        for loop_prediction, ridge_prediction in zip(loop_predictions, ridge_predictions):
            combined_prediction = loop_prediction
            combined_prediction[0] = combined_prediction[0] - ridge_prediction / 3
            combined_prediction[1] = combined_prediction[1] - ridge_prediction * 2 / 3
            combined_prediction[2] = combined_prediction[2] - ridge_prediction
            combined_prediction[3:] = combined_prediction[3:] - ridge_prediction
            combined_predictions.append(combined_prediction)

        # Recursively add the errors
        for i in range(2, self.prediction_horizon // 5 - 3, 3):
            suffix = '_what_if_' + str((i + 1) * 5)
            current_features = [feature + suffix for feature in self.ridge_features
                                if not feature.startswith('interaction') if not feature.startswith('CGM')]
            current_df = x_test.copy()[current_features]

            # Remove the "_what_if_" suffix from column names
            current_df.rename(columns=lambda x: x[:-len(suffix)] if x.endswith(suffix) else x, inplace=True)

            # Add "cgm" but from predicted value previous iteration
            current_df['CGM'] = [prediction[i] for prediction in combined_predictions]

            # Standardize and add interaction terms
            current_df = self.add_interaction_terms(current_df)

            ridge_predictions = self.ridge_model.best_estimator_.predict(current_df[self.ridge_features])

            for combined_prediction, ridge_prediction in zip(combined_predictions, ridge_predictions):
                combined_prediction[i + 1] = combined_prediction[i + 1] - ridge_prediction / 3
                combined_prediction[i + 2] = combined_prediction[i + 2] - ridge_prediction * 2 / 3
                combined_prediction[i + 3] = combined_prediction[i + 3] - ridge_prediction
                combined_prediction[i + 3:] = combined_prediction[i + 3:] - ridge_prediction

        return combined_predictions

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return self.ridge_model.best_estimator_.alpha

    def add_interaction_terms(self, df):
        # Scale the numerical features
        X = df[self.transformed_features].values
        X_scaled = self.standard_scaler.transform(X)
        df_scaled = pd.DataFrame(X_scaled, index=df.index, columns=self.transformed_features)
        df[self.transformed_features] = df_scaled

        for col in [val for val in self.ridge_features if val.startswith('hour')]:
            df[col] = df[col] * df['active_insulin']

        # Calculate interaction terms
        df['interaction_CGM_carbs'] = df['CGM'] * df['active_carbs']
        df['interaction_CGM_insulin'] = df['CGM'] * df['active_insulin']

        # TODO: (Wait until exposed to be needed:) Calculate long term effects, correlation hour of day...

        return df

