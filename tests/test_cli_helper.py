import pytest
from glupredkit.helpers.cli import (split_string, validate_config_file_name, validate_subject_ids, validate_prediction_horizon,
                                    validate_num_lagged_features, validate_feature_list, validate_test_size)
from click import BadParameter


def test_split_string():
    assert split_string("a, b, c") == ["a", "b", "c"]
    assert split_string("") == []
    assert split_string(None) == []


def test_validate_config_file_name():
    # Mock objects for ctx and param, which are not used in the function
    mock_ctx = None
    mock_param = None

    # Test cases for various file name inputs
    test_cases = [
        ("filename.txt", "filename"),
        ("filename", "filename"),
        ("filename.with.multiple.dots.txt", "filename.with.multiple.dots"),
        ("filename.with.multiple.dots", "filename.with.multiple"),
        ("filename/with/path/filename.json", "filename"),
        ("filename.", "filename"),
        (".hiddenfile", ".hiddenfile"),
        ("", ""),
        (123, "123"),  # Non-string input
        (None, "None")  # None as a string
    ]

    for input_value, expected_output in test_cases:
        assert validate_config_file_name(mock_ctx, mock_param, input_value) == expected_output, f"Failed on input {input_value}"


# Mock objects for ctx and param, which are not used in the functions
mock_ctx = None
mock_param = None


def test_validate_subject_ids():
    assert validate_subject_ids(mock_ctx, mock_param, None) == []
    assert validate_subject_ids(mock_ctx, mock_param, '') == []
    assert validate_subject_ids(mock_ctx, mock_param, '1,2,3') == (1, 2, 3)
    with pytest.raises(BadParameter):
        validate_subject_ids(mock_ctx, mock_param, '1,a,3')


def test_validate_prediction_horizon():
    assert validate_prediction_horizon(mock_ctx, mock_param, '10') == 10
    assert validate_prediction_horizon(mock_ctx, mock_param, 10) == 10
    with pytest.raises(BadParameter):
        validate_prediction_horizon(mock_ctx, mock_param, '-5')
    with pytest.raises(BadParameter):
        validate_prediction_horizon(mock_ctx, mock_param, 'not_an_integer')


def test_validate_num_lagged_features():
    assert validate_num_lagged_features(mock_ctx, mock_param, '5') == 5
    assert validate_num_lagged_features(mock_ctx, mock_param, 5) == 5
    with pytest.raises(BadParameter):
        validate_num_lagged_features(mock_ctx, mock_param, '-5')
    with pytest.raises(BadParameter):
        validate_num_lagged_features(mock_ctx, mock_param, 'not_an_integer')


def test_validate_feature_list():
    assert validate_feature_list(mock_ctx, mock_param, '') == []
    assert validate_feature_list(mock_ctx, mock_param, '1, 2, 3') == ['1', '2', '3']
    assert validate_feature_list(mock_ctx, mock_param, '[1,2,3]') == [1, 2, 3]
    assert validate_feature_list(mock_ctx, mock_param, 'not_a_list') == ['not_a_list']
    assert validate_feature_list(mock_ctx, mock_param, '1, 2, a') == ['1', '2', 'a']


def test_validate_test_size():
    assert validate_test_size(mock_ctx, mock_param, '0.5') == 0.5
    assert validate_test_size(mock_ctx, mock_param, 0.5) == 0.5
    with pytest.raises(BadParameter):
        validate_test_size(mock_ctx, mock_param, '-0.5')
    with pytest.raises(BadParameter):
        validate_test_size(mock_ctx, mock_param, '1.5')
    with pytest.raises(BadParameter):
        validate_test_size(mock_ctx, mock_param, 'not_a_float')


