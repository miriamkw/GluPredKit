import os
import pytest
import shutil
from click.testing import CliRunner
from glupredkit.cli import setup_directories, parse, generate_config


@pytest.fixture(scope="session")
def runner():
    return CliRunner()


@pytest.fixture(scope="session")
def temp_dir(runner):
    test_data_dir = os.path.join('tests', 'test_data')

    with runner.isolated_filesystem(temp_dir=test_data_dir) as temp_dir:
        yield temp_dir

    # Clean up the directory after the test session
    # shutil.rmtree(test_data_dir)


def test_setup_directories(runner, temp_dir):
    # Run the setup_directories command
    result = runner.invoke(setup_directories)

    # Check that the command ran successfully
    assert result.exit_code == 0
    assert "Directories created for usage of GluPredKit." in result.output

    # Verify that the directories were created
    expected_dirs = [
        'data/raw',
        'data/configurations',
        'data/trained_models',
        'data/tested_models',
        'data/figures',
        'data/reports'
    ]
    for directory in expected_dirs:
        assert os.path.exists(directory)


# TODO: Parse data

def test_generate_config(runner, temp_dir):
    # TODO: REMOVE THIS AFTER CREATING THE PARSE DATA OR REAL MOCK DATA TEST INSTEAD!
    open('data/raw/df.csv', 'a').close()

    # Define input values for the prompts
    inputs_1 = [
        '--file-name', 'my_config_1',
        '--data', 'df.csv',
        '--prediction-horizon', '60',
        '--num-lagged-features', '12',
        '--num-features', 'CGM,insulin,carbs'
    ]
    inputs_2 = [
        '--file-name', 'my_config_2',
        '--data', 'df.csv',
        '--subject-ids', '540,544',
        '--prediction-horizon', '180',
        '--num-lagged-features', '18',
        '--num-features', 'CGM,insulin,carbs',
        '--cat-features', 'hour',
        '--what-if-features', 'insulin,carbs'
    ]
    result_1 = runner.invoke(generate_config, inputs_1)
    result_2 = runner.invoke(generate_config, inputs_2)

    # Asserts to check successful execution
    assert result_1.exit_code == 0
    assert 'Note that it might take a minute before the file appears in the folder.' in result_1.output

    assert result_2.exit_code == 0
    assert 'Note that it might take a minute before the file appears in the folder.' in result_2.output

    # Check if the files were written
    config_path_1 = os.path.join('data', 'configurations', 'my_config_1.json')
    assert os.path.exists(config_path_1)

    config_path_2 = os.path.join('data', 'configurations', 'my_config_2.json')
    assert os.path.exists(config_path_2)



# TODO: What happens if there are features in the config that are not in the dataset?
# TODO: What happens if the subject ids are not in the dataset?
# TODO: What happens if the inputs are incorrect


