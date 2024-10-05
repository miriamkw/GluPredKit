from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from glupredkit.helpers.cli import get_trained_model
import pandas as pd
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.subject_ids = None
        self.recursion_samples = None
        self.loop_model = None
        self.input_features = None
        self.models = []


    # TODO: Rename to phyisological model, "loop_model" could in theory be any model!
    def _fit_model(self, x_train, y_train, loop_model=None, recursion_samples=6, *args):
        self.subject_ids = x_train['id'].unique()
        self.recursion_samples = recursion_samples

        if not loop_model:
            raise ValueError("The hybrid model must have a loop model file as an input!")

        # 1) Load the loop model from input
        self.loop_model = get_trained_model(loop_model)
        # print("predictions:", trained_loop_model.predict(x_train.iloc[:100]))

        # 2) Process the data to get target values with errors of the loop model
        # TODO: Remove n_samples when you have a working version
        n_samples = 200

        y_pred = pd.DataFrame(self.loop_model.predict(x_train.iloc[:n_samples]), columns=y_train.columns)
        y_train_reset_index = y_train.iloc[:n_samples].reset_index(drop=True)
        y_train_loop_error = y_train_reset_index - y_pred

        # 3) Train a model for some (adjustable) recursive interval
        # TODO: Add features like IOB, COB, CGM delta, hour of day...
        x_train['CGM_diff_5min'] = x_train['CGM'] - x_train['CGM_5']
        x_train['CGM_diff_10min'] = x_train['CGM'] - x_train['CGM_10']
        x_train['CGM_diff_15min'] = x_train['CGM'] - x_train['CGM_15']
        x_train['CGM_diff_30min'] = x_train['CGM'] - x_train['CGM_30']

        time_lagged_samples = 3
        features = []
        control_features = ['exercise', 'carbs', 'bolus', 'basal']
        columns = features + control_features + ['CGM_diff_5min', 'CGM_diff_10min', 'CGM_diff_15min', 'CGM_diff_30min']
        for i in range(time_lagged_samples):
            for feature in features + control_features:
                columns += [f'{feature}_{int((i+1)*5)}']
        for i in range(recursion_samples):
            for feature in control_features:
                columns += [f'{feature}_what_if_{int((i+1)*5)}']

        self.input_features = columns
        x_train_loop_error = x_train[columns].iloc[:n_samples]

        # TODO:4) implement the prediction adding the error recursively
        # TODO:5) add different alternatives for model approaches

        # Define the parameter grid
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0]
        }
        for _ in range(len(self.subject_ids)):
            # TODO: Add a pipeline with standardscaler and one hot encoding!
            numerical_features = x_train_loop_error.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = x_train_loop_error.select_dtypes(include=['object', 'category']).columns

            print("NUMERICAL FEATURES", numerical_features)
            print("CATEGORICAL FEATURES", categorical_features)

            base_regressor = Ridge(tol=1)
            # TODO: Should we use another scorer? (but remember that the prediction is error, not BG)
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

            # PRINT TARGETS TO SEE HOW MUCH ERROR THERE USUALLY IS
            print("ERROR VALUES", y_train_loop_error[f'target_{recursion_samples*5}'])
            print("MEAN ERROR", np.mean(y_train_loop_error[f'target_{recursion_samples*5}']))

            model_pipeline.fit(x_train_loop_error, y_train_loop_error[f'target_{recursion_samples*5}'])



            # PRINTING HOW WELL IT PERFORMED ON THE TRAINING DATA
            y_train_pred = model_pipeline.predict(x_train_loop_error)
            print("Y TRAIN PRED", y_train_pred)


            mse = mean_squared_error(y_train_loop_error[f'target_{recursion_samples * 5}'], y_train_pred)
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            print(f"Training RMSE: {rmse}")
            target_std = np.std(y_train_loop_error[f'target_{recursion_samples * 5}'])
            print(f"Target Standard Deviation: {target_std}")

            # Compare the RMSE to the variability of the target values
            print(f"RMSE as a percentage of target variability: {(rmse / target_std) * 100:.2f}%")





            self.models = self.models + [model_pipeline]
        return self

    def _predict_model(self, x_test):

        y_pred = []

        num_samples_while_debugging = 1

        n_recursions = int(self.prediction_horizon // 5 / self.recursion_samples)

        for subject_idx, subject_id in enumerate(self.subject_ids):
            df_subset = x_test.copy()[x_test['id'] == subject_id]

            loop_model_predictions = self.loop_model.predict(df_subset)
            print("LOOP MODEL PREDICTIONS: ", loop_model_predictions)

            df_subset['CGM_diff_5min'] = df_subset['CGM'] - df_subset['CGM_5']
            df_subset['CGM_diff_10min'] = df_subset['CGM'] - df_subset['CGM_10']
            df_subset['CGM_diff_15min'] = df_subset['CGM'] - df_subset['CGM_15']
            df_subset['CGM_diff_30min'] = df_subset['CGM'] - df_subset['CGM_30']

            error_predictions = self.models[subject_idx].named_steps['model'].best_estimator_.predict(df_subset[self.input_features])
            print("ERROR PREDICTIONS: ", error_predictions)

            #self.print_coefficients()
            print("MEAN ERROR PREDICTIONS: ", np.mean(error_predictions))

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

            # Now updated_predictions contains the predictions with errors distributed
            #print("Updated LOOP MODEL PREDICTIONS: ", updated_predictions)

            # TODO: Create a function that shifts the what-if and time-lagged features the steps of the recursion samples
            # TODO: In the first round, just do one step of recursion

            # TODO: Then go through adding the predicted error by the regressor
            y_pred += updated_predictions
        y_pred = np.array(y_pred)

        return y_pred

    def best_params(self):
        best_params = []

        for model in self.models:
            # Access the best estimator from GridSearchCV
            best_regressor = model.named_steps['model'].best_estimator_
            best_params = best_params + [best_regressor.alpha]

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
