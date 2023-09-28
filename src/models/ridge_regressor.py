from sklearn.linear_model import Ridge, Lasso
from src.models.base_model import BaseModel
from src.processors.resampler import filter_for_output_offset
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.ensemble import BaggingRegressor


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


# TODO: Merge models together, using "bootstrap aggregating"/"bagging". Then you can change between inverting and not inverting features
# TODO: Add correlation effect between activity state and insulin (should we calculate IOB to do this?)
# TODO: Add correlation effect between CGM and insulin (should we calculate IOB to do this?)
class RidgeRegressor(BaseModel):
    def __init__(self, is_constrained=True, interval=30, prediction_horizon=60*3):
        """
        is_constrained -- whether the coefficients should be constrained
        interval -- the interval in minutes between each prediction in the predicted trajectory. Assuming constant.
        prediction_horizon -- the total amount of minutes into the future of the last prediction
        """
        self.models = []

        self.is_constrained = is_constrained
        self.interval = interval
        self.prediction_horizon = prediction_horizon

    def fit(self, df):
        """
        df -- The dataframe produced after preprocessing using the resampler
        """
        for n in range(self.interval, self.prediction_horizon + 1, self.interval):
            X, y = filter_for_output_offset(n, df)
            X = self.invert_features(X)

            ct = get_column_transformer(X)

            model = make_pipeline(ct, Ridge(positive=self.is_constrained))

            # Define the range of 'alpha' values you want to try
            param_grid = {'ridge__alpha': [0.1, 1.0, 10.0, 100.0, 200.0, 500.0]}  # Add more values if needed
            #param_grid = {'ridge__alpha': [1000.0]}

            # Create the GridSearchCV object
            grid_search = GridSearchCV(model, param_grid, cv=5)  # cv specifies the number of cross-validation folds

            # Fit the GridSearchCV to your data
            grid_search.fit(X, y)

            # Get the best estimator (with the best hyperparameter 'alpha')
            best_model = grid_search.best_estimator_

            # You can also access the best hyperparameter value directly
            best_alpha = grid_search.best_params_['ridge__alpha']

            # Print the best hyperparameter value
            print("Best 'alpha' value:", best_alpha)

            # OLD: Solution without grid search
            # model.fit(X, y)
            # self.models = self.models + [model]
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
            y_pred = [[] for el in df['CGM']]

        df = self.invert_features(df.copy())

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

    def invert_features(self, X):
        """
        Here, we invert the sign of the features that have a negative effect on blood glucose.
        We do this because we can make a constraint to force coefficients to become strictly positive.
        When using this, we can ensure that the dynamics of the inputs are properly captured in the model.
        """
        # Get feature names after preprocessing from column transformer
        feature_names = X.columns

        # Invert the sign of relevant columns
        for index, col in enumerate(feature_names):
            if col.startswith("activity_state"):
                X[col] = -X[col]
            elif col.startswith("insulin"):
                X[col] = -X[col]
            elif col.startswith("sleep_score"):
                X[col] = -X[col]
            elif col.startswith("readiness_score"):
                X[col] = -X[col]
            elif col.startswith("CGM_12h_hypo_count"):
                X[col] = -X[col]

        """
        if self.is_constrained:
            # For the features where we don't know whether it should be positive or negative,
            # we  duplicate the feature and invert it
            duplicate_cols = [el for el in X.columns if el.startswith("CGM") and len(el.split("_")) <= 2]
            duplicate_cols = duplicate_cols + [el for el in X.columns if el.startswith("CGM_what_if")]

            for index, col in enumerate(duplicate_cols):
                inverted_column_df = pd.DataFrame({col + "_inverted": -X[col]}, index=X.index)
                X = pd.concat([X, inverted_column_df], axis=1)
        """


        return X
