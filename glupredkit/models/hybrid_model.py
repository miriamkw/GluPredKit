from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from glupredkit.helpers.cli import get_trained_model
import pandas as pd
import numpy as np
import itertools


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.subject_ids = None
        self.recursion_samples = None
        self.base_model = None
        self.input_features = None
        self.models = []


    def _fit_model(self, x_train, y_train, base_model=None, recursion_samples=6, ml_model='ridge', *args):
        self.subject_ids = x_train['id'].unique()
        self.recursion_samples = recursion_samples

        if not base_model:
            raise ValueError("The hybrid model must have a base model file as an input!")

        # 1) Load the loop model from input
        self.base_model = get_trained_model(base_model)

        # 2) Process the data to get target values with errors of the loop model
        # TODO: Remove n_samples when you have a working version
        n_samples = 10000

        y_pred = pd.DataFrame(self.base_model.predict(x_train.iloc[-n_samples:]), columns=y_train.columns)
        y_train_reset_index = y_train.iloc[-n_samples:].reset_index(drop=True)
        y_train_loop_error = y_train_reset_index - y_pred

        # 3) Train a model for some (adjustable) recursive interval
        x_train, columns = self.add_columns_and_get_feature_names(x_train)

        self.input_features = columns
        x_train_loop_error = x_train[columns].iloc[-n_samples:]

        # Define the parameter grid
        if ml_model == 'ridge':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
            base_regressor = Ridge(tol=1)
        elif ml_model == 'lasso':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
            base_regressor = Lasso(tol=1)
        elif ml_model == 'elasticnet':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.2, 0.5, 0.8]  # Adding L1/L2 mix for ElasticNet
            }
            base_regressor = ElasticNet(tol=1)
        elif ml_model == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
            base_regressor = RandomForestRegressor()
        elif ml_model == 'svr':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
            base_regressor = SVR()
        elif ml_model == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],  # Number of boosting stages
                'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
                'max_depth': [3, 5, 7],  # Maximum depth of the individual estimators
                'subsample': [0.8, 1.0]  # Fraction of samples used for fitting individual base learners
            }
            base_regressor = GradientBoostingRegressor()
        else:
            raise ValueError(
                f"Model '{ml_model}' is not recognized. Please choose from: 'ridge', 'lasso', 'elasticnet', "
                f"'random_forest', 'gradient_boosting', 'svr'.")

        for _ in range(len(self.subject_ids)):
            numerical_features = x_train_loop_error.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = x_train_loop_error.select_dtypes(include=['object', 'category']).columns

            print("NUMERICAL FEATURES", numerical_features)
            print("CATEGORICAL FEATURES", categorical_features)

            model = GridSearchCV(base_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')

            # Define the preprocessing pipeline
            preprocessor = ColumnTransformer(transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

            # Define the full pipeline, including preprocessing and the model
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)  # Replace with your actual model
            ])

            model_pipeline.fit(x_train_loop_error, y_train_loop_error[f'target_{recursion_samples*5}'])

            # Printing the prediction accuracy on the training data
            y_train_pred = model_pipeline.predict(x_train_loop_error)
            mse = mean_squared_error(y_train_loop_error[f'target_{recursion_samples * 5}'], y_train_pred)
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            print(f"Training RMSE: {rmse}")
            target_std = np.std(y_train_loop_error[f'target_{recursion_samples * 5}'])
            print(f"Target Standard Deviation: {target_std}")

            # Compare the RMSE to the variability of the target values
            print(f"RMSE as a percentage of target variability: {(rmse / target_std) * 100:.2f}%")

            self.models = self.models + [model_pipeline]

        #self.print_coefficients()
        return self

    def _predict_model(self, x_test):

        y_pred = []

        #num_samples_while_debugging = 1

        # A factor to reduce the impact of the predicted error
        error_factor = 1.0

        n_recursions = int(self.prediction_horizon // 5 / self.recursion_samples)

        for subject_idx, subject_id in enumerate(self.subject_ids):
            df_subset = x_test.copy()[x_test['id'] == subject_id]#.iloc[:num_samples_while_debugging]

            # TODO: Use the evaluation report instead of recomputing every time
            loop_model_predictions = self.base_model.predict(df_subset)
            #print("LOOP MODEL PREDICTIONS: ", loop_model_predictions)

            df_subset, _ = self.add_columns_and_get_feature_names(df_subset)
            error_predictions = self.models[subject_idx].predict(df_subset[self.input_features]) * error_factor
            #print("ERROR PREDICTIONS: ", error_predictions)

            #self.print_coefficients()
            #print("MEAN ERROR PREDICTIONS: ", np.mean(error_predictions))

            updated_predictions = []
            for i, pred_list in enumerate(loop_model_predictions):
                updated_pred_list = pred_list.copy()
                total_samples = len(updated_pred_list)

                # Add the corresponding error to all values in the updated prediction list
                for j in range(total_samples):
                    # Calculate the proportion of the error to add based on the recursion sample
                    if j < self.recursion_samples:
                        # For the first recursion_samples, apply the error proportionally
                        updated_pred_list[j] += error_predictions[i] * (j + 1) / self.recursion_samples
                    else:
                        # After recursion_samples, simply add the error
                        updated_pred_list[j] += error_predictions[i]

                # Append the updated prediction list to the updated_predictions
                updated_predictions.append(updated_pred_list)

            #print(f"PRED AFTER FIRST ERROR: {updated_predictions}")

            # Recursively add predictions
            for nth_recursion in range(1, n_recursions):

                updated_df = self.get_updated_df(df_subset, nth_recursion, updated_predictions)
                #print(f"DF AFTER RECURSION {nth_recursion}", updated_df[analysis_cols])

                error_predictions = self.models[subject_idx].predict(updated_df[self.input_features]) * error_factor
                #print(f"ERROR PREDICTIONS REC {nth_recursion}: ", error_predictions)

                prev_updated_predictions = updated_predictions.copy()
                updated_predictions = []

                # Update predictions one by one, for predictions that are already updated once
                for i, pred_list in enumerate(prev_updated_predictions):
                    updated_pred_list = pred_list.copy()
                    total_samples = len(updated_pred_list)

                    # Add the corresponding error to all values in the updated prediction list
                    for j in range(nth_recursion * self.recursion_samples,total_samples):
                        # Calculate the proportion of the error to add based on the recursion sample
                        if j < nth_recursion * self.recursion_samples + self.recursion_samples:
                            # For the first recursion_samples, apply the error proportionally
                            updated_pred_list[j] += error_predictions[i] * (j + 1 - nth_recursion * self.recursion_samples) / self.recursion_samples
                        else:
                            # After recursion_samples, simply add the error
                            updated_pred_list[j] += error_predictions[i]

                    # Append the updated prediction list to the updated_predictions
                    updated_predictions.append(updated_pred_list)

                #print(f"PRED AFTER REC {nth_recursion}: {updated_predictions}")

            # Now updated_predictions contains the predictions with errors distributed
            #print("FINAL LOOP MODEL PREDICTIONS: ", updated_predictions)

            y_pred += updated_predictions
        y_pred = np.array(y_pred)

        return y_pred

    def best_params(self):
        best_params = []

        for model in self.models:
            # Access the best estimator from GridSearchCV
            best_regressor = model.named_steps['model'].best_estimator_
            best_params = best_params + [best_regressor]

        # Return the best parameters found by GridSearchCV
        return best_params

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def print_coefficients(self):
        for model in self.models:
            # Access the best estimator from GridSearchCV
            best_regressor = model.named_steps['model'].best_estimator_
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            coefficients = best_regressor.coef_
            for feature_name, coefficient in zip(feature_names, coefficients):
                print(f"Feature: {feature_name}, Coefficient: {coefficient:.4f}")


    def add_columns_and_get_feature_names(self, df):
        #print("Adding columns")
        # TODO: Make it dynamic so that if some of the features not available, we skip them
        df['CGM_diff_5min'] = df['CGM'] - df['CGM_5']
        df['CGM_diff_10min'] = df['CGM'] - df['CGM_10']
        df['CGM_diff_15min'] = df['CGM'] - df['CGM_15']
        df['CGM_diff_30min'] = df['CGM'] - df['CGM_30']

        time_lagged_samples = 6
        features = ['CGM']
        #control_features = ['exercise', 'carbs', 'bolus', 'basal', 'active_insulin', 'active_carbs']
        control_features = ['exercise', 'active_insulin', 'active_carbs']
        #control_features = ['exercise', 'carbs', 'bolus', 'basal']
        columns = features + control_features + ['CGM_diff_5min', 'CGM_diff_10min', 'CGM_diff_15min', 'CGM_diff_30min']
        for i in range(time_lagged_samples):
            for feature in features + control_features:
                columns += [f'{feature}_{int((i+1)*5)}']
        for i in range(self.recursion_samples):
            for feature in control_features:
                columns += [f'{feature}_what_if_{int((i+1)*5)}']

        # Adding interaction effects
        interaction_features = ['CGM', 'exercise', 'active_insulin', 'active_carbs']
        #interaction_features = ['CGM', 'exercise']
        for feature1, feature2 in itertools.combinations(interaction_features, 2):
            interaction_name = f'{feature1}_x_{feature2}'
            df[interaction_name] = df[feature1] * df[feature2]
            columns += [interaction_name]

        for i in range(time_lagged_samples):
            time_lag = (i+1)*5
            interaction_features = [f'CGM_{time_lag}', f'exercise_{time_lag}', f'active_insulin_{time_lag}', f'active_carbs_{time_lag}']
            #interaction_features = [f'CGM_{time_lag}', f'exercise_{time_lag}']

            for feature1, feature2 in itertools.combinations(interaction_features, 2):
                interaction_name = f'{feature1}_x_{feature2}'
                df[interaction_name] = df[feature1] * df[feature2]
                columns += [interaction_name]

        for i in range(self.recursion_samples):
            what_if_lag = (i+1)*5
            interaction_features = ['CGM', f'exercise_what_if_{what_if_lag}', f'active_insulin_what_if_{what_if_lag}', f'active_carbs_what_if_{what_if_lag}']
            #interaction_features = ['CGM', f'exercise_what_if_{what_if_lag}']

            for feature1, feature2 in itertools.combinations(interaction_features, 2):
                interaction_name = f'{feature1}_x_{feature2}'
                df[interaction_name] = df[feature1] * df[feature2]
                columns += [interaction_name]

        # Adding categorical features. Convert to string so it can be automatically detected as categorical feature
        df['hour'] = df.index.hour.astype(str)
        columns += ['hour']

        return df, columns


    def get_updated_df(self, original_df, nth_recursion, updated_predictions):
        """
        Skew the features for the recursive predictions
        """
        df_updated = original_df.copy()

        time_lagged_samples = 6
        time_lag = int(nth_recursion * self.recursion_samples * 5)

        df_updated[f'CGM'] = [pred[nth_recursion * self.recursion_samples - 1] for pred in updated_predictions]
        for i in range(time_lagged_samples):
            pred_index = nth_recursion * self.recursion_samples - i - 2
            if pred_index >= 0:
                df_updated[f'CGM_{(i+1)*5}'] = [val[pred_index] for val in updated_predictions]
            else:
                pred_index = np.abs(pred_index)
                if pred_index == 1:
                    df_updated[f'CGM_{(i+1)*5}'] = original_df['CGM']
                else:
                    df_updated[f'CGM_{(i+1)*5}'] = original_df[f'CGM_{(pred_index - 1) * 5}']

        features = ['exercise', 'active_insulin', 'active_carbs']

        for feature in features:
            # Update current time step
            df_updated[feature] = original_df[f'{feature}_what_if_{time_lag}']

            # Update what if events
            for i in range(self.recursion_samples):
                df_updated[f'{feature}_what_if_{(i+1)*5}'] = original_df[f'{feature}_what_if_{(i+1)*5 + time_lag}']

            # Update time lagged events (using as many time lags as in the other function)
            for i in range(time_lagged_samples):
                if (i+1)*5 == time_lag:
                    df_updated[f'{feature}_{(i + 1) * 5}'] = original_df[feature]
                elif (i+1)*5 < time_lag:
                    df_updated[f'{feature}_{(i+1)*5}'] = original_df[f'{feature}_what_if_{time_lag - (i+1)*5}']
                else:
                    df_updated[f'{feature}_{(i + 1) * 5}'] = original_df[f'{feature}_{(i + 1) * 5 - time_lag}']

        # Add the interaction effects given the updated features
        df_updated, _ = self.add_columns_and_get_feature_names(df_updated)

        # Update the hour
        time_to_add = nth_recursion * self.recursion_samples * 5
        df_updated['hour'] = (df_updated.index + pd.to_timedelta(time_to_add, unit='minutes')).hour.astype(str)

        return df_updated


