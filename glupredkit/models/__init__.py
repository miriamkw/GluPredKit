import importlib


def safe_import(module_name, class_name, package="glupredkit.models"):
    """Attempts to import a module and return its class. If the module is missing, raises ImportError when accessed."""
    try:
        module = importlib.import_module(module_name, package=package)
        return getattr(module, class_name)
    except ModuleNotFoundError:
        return None  # Return None instead of raising an error


DoubleLSTM = safe_import(".double_lstm", "Model")
Loop = safe_import(".loop", "Model")
LoopV2 = safe_import(".loop_v2", "Model")
LSTM = safe_import(".lstm", "Model")
MTL = safe_import(".mtl", "Model")
NaiveLinearRegressor = safe_import(".naive_linear_regressor", "Model")
RandomForest = safe_import(".random_forest", "Model")
Ridge = safe_import(".ridge", "Model")
StackedPLSR = safe_import(".stacked_plsr", "Model")
STL = safe_import(".stl", "Model")
SVR = safe_import(".svr", "Model")
TCN = safe_import(".tcn", "Model")
UvaPadova = safe_import(".uva_padova", "Model")
WeightedRidge = safe_import(".weighted_ridge", "Model")
ZeroOrderHold = safe_import(".zero_order", "Model")
