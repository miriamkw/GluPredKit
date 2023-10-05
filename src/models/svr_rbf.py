from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from .base_model import BaseModel


# Function to extend feature_names with time-lagged features
def extend_feature_names(feature_names, df):
    extended_feature_names = set(feature_names)
    for col in df.columns:
        for feature in feature_names:
            if col.startswith(feature):
                extended_feature_names.add(col)
    return list(extended_feature_names)


class Model(BaseModel):
    def __init__(self, prediction_horizon, numerical_features, categorical_features):
        super().__init__(prediction_horizon, numerical_features, categorical_features)

        self.model = None

    def fit(self, x_train, y_train):
        # Perform grid search to find the best parameters and fit the model

        # Extend feature names to include time-lagged features
        self.numerical_features = extend_feature_names(self.numerical_features, x_train)
        self.categorical_features = extend_feature_names(self.categorical_features, x_train)

        # Define the preprocessing for numeric and categorical features
        transformers = [
            ('num', StandardScaler(), self.numerical_features),
            ('cat', OneHotEncoder(), self.categorical_features)
        ]

        # Combine all transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(transformers)

        # Define the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', SVR(tol=1, kernel='linear'))
        ])

        # Define the parameter grid
        param_grid = {
            'regressor__C': [1, 30, 50],
            'regressor__epsilon': [0.01, 0.015],
            'regressor__gamma': [0.003, 0.008],
        }

        # Define GridSearchCV
        self.model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        # Use the best estimator found by GridSearchCV to make predictions
        y_pred = self.model.best_estimator_.predict(x_test)
        return y_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return self.model.best_params_
