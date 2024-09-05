import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open
from glupredkit.helpers.model_config_manager import generate_model_configuration, ModelConfigurationManager


def test_generate_model_configuration_success():
    with patch('os.path.isfile', return_value=True), \
         patch('builtins.open', mock_open()) as mocked_file, \
         patch('builtins.__import__'):
        generate_model_configuration('test_config', 'data1', [1, 2, 3], 'standard', 30, 5, [10, 20], [], [])
        mocked_file.assert_called_once_with(Path('data/configurations/test_config.json'), 'w')


def test_generate_model_configuration_preprocessor_missing():
    with patch('os.path.isfile', return_value=True), \
         patch('builtins.__import__', side_effect=ImportError):  # Updated to correct reference
        with pytest.raises(ValueError):
            generate_model_configuration('test_config', 'data1', [1, 2, 3], 'missing_preprocessor', 30, 5, [10, 20], [], [])


def test_load_config_success():
    mock_config = {
        "data": "data.csv",
        "subject_ids": [1, 2, 3],
        "preprocessor": "standard",
        "prediction_horizon": 30,
        "num_lagged_features": 5,
        "num_features": [10, 20],
        "cat_features": [],
        "what_if_features": []
    }
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
        manager = ModelConfigurationManager('test_config')
        assert manager.get_data() == 'data.csv'


def test_load_config_file_not_found():
    with patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            ModelConfigurationManager('missing_config')


def test_config_validation_error_missing_key():
    incomplete_config = {
        "data": "data.csv",  # Missing other keys
    }
    with patch('builtins.open', mock_open(read_data=json.dumps(incomplete_config))):
        with pytest.raises(ValueError) as excinfo:
            ModelConfigurationManager('incomplete_config')
        assert "Missing key 'subject_ids' in the config file" in str(excinfo.value)
