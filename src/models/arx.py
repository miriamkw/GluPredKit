from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.model = None

    def fit(self, x_train, y_train):
        # Define the model
        pipeline = Pipeline([('regressor', LinearRegression())])

        # Define GridSearchCV
        self.model = pipeline
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        # Use the best estimator found by GridSearchCV to make predictions
        y_pred = self.model.predict(x_test)
        return y_pred

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None
