import pytest
import json
import os
from unittest import mock
from glupredkit.helpers.unit_config_manager import UnitConfigManager


@pytest.fixture
def mock_default_config():
    return json.dumps({
        "use_mgdl": True
    })


@pytest.fixture
def mock_user_config_file(tmp_path, mock_default_config):
    config_dir = tmp_path / ".glupredkit"
    config_dir.mkdir()
    config_file = config_dir / "unit_config.json"
    config_file.write_text(mock_default_config)
    return config_file


@pytest.fixture
def config_manager(mock_user_config_file, mock_default_config):
    with mock.patch('importlib.resources.files') as mock_files, \
            mock.patch('os.path.expanduser', return_value=str(mock_user_config_file.parent)):
        mock_files.return_value.__truediv__.return_value = mock_user_config_file

        manager = UnitConfigManager()
        yield manager


def test_initialization(config_manager, mock_user_config_file):
    assert config_manager.user_config_file == str(mock_user_config_file)
    assert config_manager.config == {"use_mgdl": True}


def test_initialize_user_config(tmp_path, config_manager):
    config_dir = tmp_path / ".glupredkit"
    config_file = config_dir / "unit_config.json"

    assert os.path.exists(config_manager.user_config_file)
    with open(config_file, 'r') as f:
        assert json.load(f) == {"use_mgdl": True}


def test_use_mgdl_property(config_manager):
    assert config_manager.use_mgdl is True
    config_manager.use_mgdl = False
    assert config_manager.use_mgdl is False


def test_convert_value(config_manager):
    config_manager.use_mgdl = True
    assert config_manager.convert_value(180) == 180
    config_manager.use_mgdl = False
    assert pytest.approx(config_manager.convert_value(180), 0.001) == 9.991


def test_get_unit(config_manager):
    assert config_manager.get_unit() == "mg/dL"
    config_manager.use_mgdl = False
    assert config_manager.get_unit() == "mmol/L"
