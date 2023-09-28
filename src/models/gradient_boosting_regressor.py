from src.models.base_model import BaseModel
from src.processors.resampler import filter_for_output_offset
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor as gb


def get_column_transformer(X):
    categorical_features = ['hour']
    numeric_features = [el for el in X.columns if el not in categorical_features]
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder()),
        ]
    )
    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return ct


class GradientBoostingRegressor(BaseModel):
    def __init__(self, interval=30, prediction_horizon=60*3):
        """
        is_constrained -- whether the coefficients should be constrained
        interval -- the interval in minutes between each prediction in the predicted trajectory. Assuming constant.
        prediction_horizon -- the total amount of minutes into the future of the last prediction
        """
        self.models = []

        self.interval = interval
        self.prediction_horizon = prediction_horizon

    def fit(self, df):
        """
        df -- The dataframe produced after preprocessing using the resampler
        """
        for n in range(self.interval, self.prediction_horizon + 1, self.interval):
            X, y = filter_for_output_offset(n, df)

            ct = get_column_transformer(X)

            model = make_pipeline(ct, gb())

            # Define the range of hyperparameter values you want to try
            param_grid = {
                #'gradientboostingregressor__n_estimators': [50, 100, 150],
                #'gradientboostingregressor__learning_rate': [0.1, 0.01, 0.001],
                #'gradientboostingregressor__max_depth': [1, 3, 5, 7],
                'gradientboostingregressor__min_samples_split': [2, 3, 5, 7],
                'gradientboostingregressor__min_samples_leaf': [1, 3, 5, 7]
            }

            # Create the GridSearchCV object
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')  # cv specifies the number of cross-validation folds

            # Fit the GridSearchCV to your data
            grid_search.fit(X, y)

            # Get the best estimator (with the best hyperparameter 'alpha')
            best_model = grid_search.best_estimator_

            # You can also access the best hyperparameter value directly
            best_params = grid_search.best_params_

            # Print the best hyperparameter value
            print("Best params:", best_params)

            self.models = self.models + [best_model]

        return self


    def predict(self, df, include_measurements=True):
        """
        df -- The dataframe produced after preprocessing using the resampler
        include_measurements -- Whether to include the reference measurement in the predicted trajectories

        Output -- a list of predicted trajectories (list of lists)
        """
        if include_measurements:
            y_pred = [[el] for el in df['CGM']]
        else:
            y_pred = [[] for _ in df['CGM']]

        i = self.interval

        for model in self.models:
            predictions = model.predict(df)

            # Add the prediction to the CGM what if events
            cgm_what_if_col_name = 'CGM_what_if_' + str(i)
            df[cgm_what_if_col_name] = predictions
            i = i + self.interval

            for index, el in enumerate(y_pred):
                y_pred[index] = y_pred[index] + [predictions[index]]

        return y_pred
