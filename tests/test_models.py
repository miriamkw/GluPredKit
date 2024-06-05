import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from glupredkit.models.loop import Model as Loop
from glupredkit.models.naive_linear_regressor import Model as NaiveLinearRegressor
from glupredkit.models.random_forest import Model as RandomForest
from glupredkit.models.ridge import Model as Ridge
from glupredkit.models.svr import Model as SVR
from glupredkit.models.tcn import Model as TCN
from glupredkit.models.uva_padova import Model as UvaPadova
from glupredkit.models.zero_order import Model as ZeroOrder

# Defining the list of model classes
model_classes = [
    Loop,
    NaiveLinearRegressor,
    RandomForest,
    Ridge,
    SVR,
    TCN,
    UvaPadova,
    ZeroOrder
]


@pytest.fixture
def sample_data():
    # Creating a sample DataFrame with necessary columns and data types
    data = {
        'id': np.tile(np.arange(1, 4), 1000),  # 3 unique IDs, repeated 10 times each
        'CGM': np.random.random(3000),
        'insulin': np.random.random(3000),
        'carbs': np.random.random(3000)
    }
    x_test = pd.DataFrame(data)
    return x_test


@pytest.mark.parametrize("model_cls", model_classes)
def test_metric_class_name(model_cls):
    model = model_cls(prediction_horizon=30)
    assert model.__class__.__name__ == "Model", f"Class name for {model_cls.__name__} is not 'Model'"


@pytest.mark.parametrize("model_cls", model_classes)
def test_predict_unfitted_model(model_cls, sample_data):
    model = model_cls(prediction_horizon=30)
    x_test = sample_data
    with pytest.raises(NotFittedError):
        model.predict(x_test)

