import os
import pytest
import numpy as np
import pandas as pd
import shutil
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

from pathlib import Path
from click.testing import CliRunner
from glupredkit.cli import (setup_directories, generate_config, train_model, evaluate_model, generate_evaluation_pdf,
                            generate_comparison_pdf, draw_plots)


@pytest.fixture(scope="session")
def runner():
    return CliRunner()


@pytest.fixture(scope="session")
def temp_dir(runner):
    """
    test_data_dir = os.path.join('tests', 'test_data')

    # Create the test_data directory if it doesn't exist
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    with runner.isolated_filesystem(temp_dir=test_data_dir) as temp_dir:
        yield temp_dir

    # Clean up the directory after the test session
    shutil.rmtree(test_data_dir)
    """
    test_data_dir = Path('tests') / 'test_data'

    # Create the test_data directory if it doesn't exist
    if not test_data_dir.exists():
        test_data_dir.mkdir(parents=True, exist_ok=True)

    with runner.isolated_filesystem(temp_dir=str(test_data_dir)) as temp_dir:
        yield temp_dir

    # Clean up the directory after the test session
    shutil.rmtree(test_data_dir)



def sample_data():
    # Define the index starting from '2024-01-01 00:00:00' with 5-minute intervals
    index = pd.date_range(start='2024-01-01', periods=30000, freq='5min')

    # Creating a sample DataFrame with necessary columns and data types
    data = {
        'id': np.repeat(np.arange(1, 4), 10000),  # 1s, then 2s, then 3s, each repeated 10000 times
        'CGM': np.random.uniform(30, 540, 30000),
        'insulin': np.random.uniform(0, 20, 30000),
        'bolus': np.random.uniform(0, 20, 30000),
        'basal': np.random.uniform(0, 2, 30000),
        'carbs': np.random.uniform(0, 150, 30000),
        'is_test': np.tile(np.repeat([False, True], 5000), 3)  # Alternate blocks of 5000
    }
    x_test = pd.DataFrame(data, index=index)

    # Rename the index column to "date"
    x_test.index.name = 'date'

    return x_test


def test_setup_directories(runner, temp_dir):
    # Run the setup_directories command
    result = runner.invoke(setup_directories)

    # Check that the command ran successfully
    assert result.exit_code == 0
    assert "Directories created for usage of GluPredKit." in result.output

    # Verify that the directories were created
    expected_dirs = [
        'raw',
        'configurations',
        'trained_models',
        'tested_models',
        'figures',
        'reports'
    ]
    for directory in expected_dirs:
        file_path = Path('data') / directory
        assert os.path.exists(file_path)


# TODO: Parse data
def test_generate_config(runner, temp_dir):
    file_path = Path('data') / 'raw' / 'df.csv'
    sample_data().to_csv(file_path)

    # Define input values for the prompts
    inputs_1 = [
        '--file-name', 'my_config_1',
        '--data', 'df.csv',
        '--prediction-horizon', '60',
        '--num-lagged-features', '12',
        '--num-features', 'CGM,basal,bolus,carbs'
    ]
    inputs_2 = [
        '--file-name', 'my_config_2',
        '--data', 'df.csv',
        '--subject-ids', '1,2',
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


def test_train_model(runner, temp_dir):
    # Define the input arguments and options
    epochs = str(2)
    config_file_name = 'my_config_1'

    args_list = [
        # ['blstm', config_file_name, '--epochs', epochs],
        # ['loop', config_file_name, '--n-cross-val-samples', 10],
        # ['lstm', config_file_name, '--epochs', epochs],
        # ['mtl', config_file_name, '--epochs', epochs],
        ['naive_linear_regressor', config_file_name],
        # ['random_forest', config_file_name],
        ['ridge', config_file_name],
        # ['stl', config_file_name, '--epochs', epochs],
        # ['tcn', config_file_name, '--epochs', epochs],
        # ['uva_padova', config_file_name, '--n-steps', 100, '--training-samples-per-subject', 100],
        ['zero_order', config_file_name]
    ]

    for args in args_list:
        # Run the command
        result = runner.invoke(
            train_model, args
        )

        # Asserts to check successful execution
        assert result.exit_code == 0
        assert "Training data finished preprocessing..." in result.output
        assert "Training model..." in result.output

        # Check if the model file was created
        output_file_name = f'{args[0]}__{config_file_name}__60.pkl'
        output_path = Path('data') / 'trained_models' / output_file_name
        assert output_path.exists(), f"Expected file {output_path} was not created"


def test_evaluate_model(runner, temp_dir):
    runner = CliRunner()

    config = 'my_config_1'
    models = ['naive_linear_regressor', 'ridge', 'zero_order']

    for model in models:

        result = runner.invoke(evaluate_model, [f'{model}__{config}__60.pkl', '--max-samples', '100'])
        assert result.exit_code == 0

        # Check if the model test file was created
        output_file_name = f'{model}__{config}__60.csv'
        output_path = Path('data') / 'tested_models' / output_file_name
        assert output_path.exists(), f"Expected file {output_path} was not created"


def test_generate_evaluation_pdf(runner, temp_dir):
    runner = CliRunner()

    config = 'my_config_1'
    models = ['naive_linear_regressor', 'ridge', 'zero_order']

    for model in models:
        result = runner.invoke(generate_evaluation_pdf, ['--results-file', f'{model}__{config}__60.csv'])
        assert result.exit_code == 0

        # Check if reports were generated
        output_file_name = f'{model}__{config}__60.pdf'
        output_path = Path('data') / 'reports' / output_file_name
        assert output_path.exists(), f"Expected file {output_path} was not created"


def test_generate_comparison_pdf(runner, temp_dir):
    runner = CliRunner()

    result = runner.invoke(generate_comparison_pdf)
    assert result.exit_code == 0


def test_draw_plots(runner, temp_dir):
    runner = CliRunner()

    config = 'my_config_1'
    results_files = f'naive_linear_regressor__{config}__60.csv,ridge__{config}__60.csv'

    result = runner.invoke(draw_plots, ['--results-files', results_files, '--plots', 'scatter_plot', '--prediction-horizons', '30'])
    assert result.exit_code == 0

