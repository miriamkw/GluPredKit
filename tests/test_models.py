import pytest
import numpy as np
import pandas as pd
import os
import json
from sklearn.exceptions import NotFittedError
from glupredkit.models.loop import Model

@pytest.fixture
def sample_data():
    # Creating a sample DataFrame with necessary columns and data types
    data = {
        'id': np.tile(np.arange(1, 4), 10),  # 3 unique IDs, repeated 10 times each
        'feature1': np.random.random(30),
        'feature2': np.random.random(30),
        # Add other features as necessary
    }
    targets = np.random.random((30, 5))  # Assuming 5 target variables
    df = pd.DataFrame(data)
    return df, targets

def test_initialization():
    model = Model(prediction_horizon=5)
    assert model.prediction_horizon == 5, "Incorrect prediction horizon set."
    assert model.models == [], "Model list should be initialized empty."

def test_fit(sample_data):
    x_train, y_train = sample_data
    model = Model(prediction_horizon=5)
    model.fit(x_train, y_train)
    assert len(model.models) == len(x_train['id'].unique()), "One model per unique 'id' should be created."
    assert all([isinstance(m.best_estimator_, MultiOutputRegressor) for m in model.models]), "Models should be fitted with MultiOutputRegressor."

def test_predict(sample_data):
    x_train, y_train = sample_data
    x_test = x_train.copy()  # Use the same data for simplicity in testing
    model = Model(prediction_horizon=5)
    model.fit(x_train, y_train)

    # Test predict functionality
    predictions = model.predict(x_test)
    assert predictions.shape[0] == x_test.shape[0], "Predictions should be generated for each sample."
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array."

def test_predict_unfitted_model(sample_data):
    x_test, _ = sample_data
    model = Model(prediction_horizon=5)
    with pytest.raises(NotFittedError):
        model.predict(x_test)


def test_save_and_load_model_weights(sample_data):
    x_train, y_train = sample_data
    model = Model(prediction_horizon=5)
    model.fit(x_train, y_train)
    temp_file = "temp_model.json"
    model.save_model_weights(temp_file)

    # Check file creation
    assert os.path.exists(temp_file), "Model weight file should be created."

    # Optionally: Load the file and check contents
    with open(temp_file, 'r') as f:
        data = json.load(f)
    assert 'coefficients' in data, "Saved JSON should include model coefficients."
    os.remove(temp_file)  # Cleanup
