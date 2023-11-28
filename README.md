# Blood Glucose Prediction-Kit

> [!NOTE]  
> **USER TESTING:** If you want to participate in user testing of GluPredKit, and help to shape the further development, send an email to miriam.k.wolff@ntnu.no. Do not hesitate with reaching out if you have any questions.

This Blood Glucose (BG) Prediction Framework streamlines the process of data handling, training, and evaluating blood 
glucose prediction models in Python. Access all features via the integrated Command Line Interface (CLI).

The figure below illustrates an overview over the pipeline including all the stages of this blood glucose prediction 
framework.

<!-- ![img.png](https://miriamkw.folk.ntnu.no/figures/pipeline_overview.png) -->
![img.png](https://miriamkw.folk.ntnu.no/figures/pipeline_overview.png)

#### GluPredKit YouTube-Tutorial

[![GluPredKit YouTube-Tutorial](https://img.youtube.com/vi/GMu_Om1gTsk/0.jpg)](https://www.youtube.com/watch?v=GMu_Om1gTsk)


## Table of Contents
1. [Setup and Installation](#setup-and-installation)
   - [Regular users: Install using pip](#regular-users-install-using-pip)
   - [Developers: Install using the cloned repository](#developers-install-using-the-cloned-repository)
2. [Usage of Command Line Interface](#usage-of-command-line-interface)
   - [Getting Started](#getting-started)
   - [Parsing Data](#parsing-data)
   - [Preprocessing Data](#preprocessing-data)
   - [Train a Model](#train-a-model)
   - [Evaluate Models](#evaluate-models)
   - [Setting Configurations](#setting-configurations)
3. [Contributing with Code](#contributing-with-code)
   - [Adding Data Source Parsers](#adding-data-source-parsers)
   - [Adding Data Preprocessors](#adding-data-preprocessors)
   - [Adding Machine Learning Prediction Models](#adding-machine-learning-prediction-models)
   - [Implementing Custom Evaluation Metrics](#implementing-custom-evaluation-metrics)
   - [Adding Evaluation Plots](#adding-evaluation-plots)
4. [Disclaimers and Limitations](#disclaimers-and-limitations)
5. [License](#license)



## Setup and Installation

To setup and install this platform, there are two options depending on whether you are a regular user or a developer:
1) **Install using pip (regular users):** If you want to access the command line interface (CLI) without reading or modifying the code, choose this option.


2) **Install using the cloned repository (developers):** This is the choice if you want to use the repository and have the code visible, and to potentially edit the code.

Choose which one is relevant for you, and follow the instructions below.


----
### Regular users: Install using pip
Open your terminal and go to an empty folder in your command line.  Note that all the data storage, trained models and results will be stored in this folder.

Creating a virtual environment is optional, but recommended. Python version 3.7, 3.8 or 3.9 is required. Create a 
virtual environment with the command `python -m venv glupredkit_venv`, and activate it with `source glupredkit_venv/bin/activate` (Mac) 
or `.glupredkit_venv\Scripts\activate` (Windows).

To set up the CLI, simply run the following command:

```
pip install glupredkit
```


----
### Developers: Install using the cloned repository

First, clone the repository and make sure you are located in the root of the directory in your command line.
To set up the repository with all requirements, simply run the following command:

```
./install.sh
```

Make sure that the virtual environment `bgp-evaluation` is activated before you proceed. If not, call `source bgp-evaluation/bin/activate`.



## Usage of Command Line Interface

The command-line tool is designed to streamline the end-to-end process of data handling, preprocessing, model training, evaluation, and configuration for blood glucose prediction. The following is a guide to using this script.

### Getting started
1) First, follow the instructions above in "Setup and Installation". 
2) Then, navigate to a desired folder in your command line. 
This is the folder where the datasets, models and results will be stored.
3) Set up the necessary directories for the GluPredKit CLI by running the following command:
   * For regular users: `glupredkit setup_directories`
   * For developers: `python -m glupredkit.cli setup_directories`

You should now have the following file structure in your desired folder:
   ```
   data/
   │
   ├── raw/
   │
   ├── configurations/
   │
   ├── trained_models/
   │
   ├── figures/
   │
   └── reports/
   ```

Now, you're ready to use the Command Line Interface (CLI) for processing and predicting blood glucose levels.


Note that the prefix for all the commands will be either `glupredkit` for regular users, or `python -m glupredkit.cli` for developers.
The `glupredkit` prefix will also work while developing, but changes made to the code will only be directly reflected when
using the `python -m glupredkit.cli` prefix.
In the examples below we will use `glupredkit`.

The following figure is an overview over all the CLI commands and how they interact with the files in the folders.
<!-- ![img.png](https://miriamkw.folk.ntnu.no/figures/CLI_Overview.png) -->
![img.png](https://miriamkw.folk.ntnu.no/figures/CLI_Overview.png)


### Parsing Data
**Description**: Parse data from a chosen source and store it as CSV in `data/raw` using the selected parser. If you have an existing dataset, you can store it in `data/raw`, skip this step and go directly to preprocessing. 
If you provide your own dataset, make sure that the dataset and all datatypes are resampled into 5-minute intervals.

```
glupredkit parse --parser [tidepool|nightscout|apple_health|ohio_t1dm] [--username USERNAME] [--password PASSWORD] [--file-path FILE_PATH] [--start-date START_DATE] [--end-date END_DATE]
```

- `--parser`: Choose a parser between `tidepool`, `nightscout`, `apple_health`, or `ohio_t1dm`.
- `--username` (Optional): Your username for the data source (for nightscout, use url).
- `--password` (Optional): Your password for the data source (for nightscout, use API-KEY).
- `--file-path`: (Optional): The file path to the raw data file that shall be parsed (required for the apple_health parser).
    - For the Ohio T1DM parser, the file path is the folder where the `test` and `train` folder are located. Example: `data/raw/`. 
- `--subject-id`: (Optional): The subject id for the data that shall be parsed (required for the Ohio T1DM parser).
- `--start-date` (Optional): Start date for data retrieval, default is two weeks ago. Format "dd-mm-yyyy".
- `--end-date` (Optional): End date for data retrieval, default is now. Format "dd-mm-yyyy".
- `--output-file-name` (Optional): The filename for the output file after parsing, without file extension.

#### Example

```
glupredkit parse --parser tidepool --username johndoe@example.com --password mypassword --start-date 01-09-2023 --end-date 30-09-2023
glupredkit parse --parser nightscout --username https://my_nightscout.net/ --password API_KEY --start-date 01-09-2023 --end-date 30-09-2023
glupredkit parse --apple_health --file-path data/raw/export.xml --start-date 01-01-2023
glupredkit parse --parser ohio_t1dm --file-path data/raw/ --subject-id 559
```

---

### Generate Model Training Configuration
**Description**: This command generates a configuration with a given raw dataset, and various settings for training blood glucose predictions. These configurations will be stored in `data/configurations/`, enabling their reuse for different model approaches and evaluations.

```
glupredkit generate_config 
```
- `--file-name`: Give a file name to the configuration. Example: `my_config`.
- `--data`: Name of the input CSV file containing the data. Note that this file needs to be located in `data/raw/`. Example: `df.csv`.
- `--preprocessor`: The name of the preprocessor that shall be used. The preprocessor must be implemented in `glupredkit/preprocessors/`. The available preprocessors are:
    - basic
    - ohio_t1dm
- `--prediction-horizons`: A comma-separated list of prediction horizons (in minutes) used in model training, without spaces. Example: `30,60`. 
- `--num-lagged-features`: The number of samples to use as time-lagged features. CGM values are sampled in 5-minute intervals, so 12 samples equals one hour.
- `--num-features`: List of numerical features, separated by comma. Note that the feature names must be identical to column names in the input file. Example: `CGM,insulin,carbs`. 
- `--cat-features`: List of categorical features, separated by comma. Note that the feature names must be identical to column names in the input file.
- `--test-size`: Test size is a number between 0 and 1, that defines the fraction of the data used for testing. Example: `0.25`. Note that for the Ohio T1DM dataset the test-size is automatically going to use the original separation between train and test data.

#### Example
Upon executing `glupredkit generate_config`, you will be sequentially prompted for each of the inputs above.



---

### Train a Model
**Description**: Train a model using the specified training data.
```
glupredkit train_model MODEL_NAME CONFIG_FILE_NAME
```
- `model`: Name of the model file (without .py) to be trained. The file name must exist in `glupredkit/models/`. The available models are:
    - arx
    - elastic_net
    - gradient_boosting_trees
    - huber
    - lasso
    - lstm
    - lstm_pytorch
    - random_forest
    - ridge
    - svr_linear
    - svr_rbf
    - tcn
    - tcn_pytorch (https://github.com/locuslab/TCN/tree/master)
- `config-file-name`: Name of the configuration to train the model (without .json). The file name must exist in `data/configurations/`.

#### Example
```
glupredkit train_model ridge my_config
```
---

### Calculate Metrics
**Description**: Evaluate one or more trained models by computing metrics (for example RMSE). The results will be stored in `results/reports`.

```
glupredkit calculate_metrics [--models MODEL_FILE_NAMES] [--metrics METRICS]
```

- `--models` (Optional): List of trained model filenames from `data/trained_models/` (with .pkl), separated by comma. Default is all the models.
- `--metrics` (Optional): List of metrics from `glupredkit/metrics/` to be computed, separated by comma. Default is RMSE. The available metrics are:
    - **Root mean squared error:** rmse
    - **Glucose-specific root mean squared error:** grmse (https://pubmed.ncbi.nlm.nih.gov/22275716/)
    - **Mean absolute error:** mae
    - **Mean error:** me
    - **Mean relative error:** mre
    - **Mean absolute relative difference:** mare
    - **Pearsons correlation coefficient:** pcc
    - **Zones in Clarke error grid:** clarke_error_grid
    - **Zones in Parkes error grid:** parkes_error_grid

#### Example
```
glupredkit calculate_metrics --models ridge_ph-60.pkl,arx_ph-60,svr_linear_ph-60.pkl --metrics rmse,mae
```
---

### Draw Plots
**Description**: This command allows users to visualize model predictions using different types of plots. It supports visualization of multiple models and can restrict the plots to certain date ranges or use artificial carbohydrate and insulin inputs for specific visualizations.

```
glupredkit draw_plots
```
- `--models`: Specify the list of trained models you'd like to visualize. Input model names separated by commas, with the ".pkl" extension. By default, all available models will be evaluated.
- `--plots`: Define the type of plots to be generated. Input the names of the plots separated by commas. If not specified, a scatter plot will be the default. The available plots are:
    - scatter_plot
    - trajectories
    - one_prediction
- `--is-real-time`: A boolean flag indicating whether to consider test data without matching true measurements. By default, it is set to False.
- `--start-date`: The start date for the predictions. If not set, the first sample from the test data will be used. Input the date in the format "dd-mm-yyyy/hh:mm".
- `--end-date`: This serves as either the end date for your range or the specific prediction date for one prediction plots. If left unspecified, the command defaults to using the last sample from the test data. The date format is "dd-mm-yyyy/hh:mm".
- `--carbs`: This allows you to set an artificial carbohydrate input for one_prediction plots. This option is only valid when is-real-time is set to True.
- `--insulin`: Similar to the carbs option, this lets you provide an artificial insulin input for one_prediction plots. Again, it's only available when is-real-time is True.

#### Example
```
glupredkit draw_plots --models ridge_ph-60.pkl,arx_ph-60,svr_linear_ph-60.pkl --plots scatter_plot --start-date 25-10-2023/14:30 --end-date 30-10-2023/16:45
```

### Real-Time Prediction Plots

**Description**: To achieve real-time predictions, it is necessary to have a data-source that provides real-time data. 
Nightscout API can be a real-time API if the insulin management system is uploading data continuously (which is, to our knowledge,
only possible to achieve with open-source insulin management systems). The steps to draw real-time plots are the following:
1. Train one or more models using the steps above.
2. Parse some up-to-date data, using for example the nightscout-parser.
3. Update the model configurations to have the data from (2) as a data source for predictions.
4. Call the `draw_plots` command with `--plots one_prediction` and `--is-real-time True` 


---
### Setting Unit of Evaluations

**Description**: Set whether to use mg/dL or mmol/L for units. You can change this after models are trained,
without retraining the models. This only has an impact on the model evaluation (`calculate_metrics` or `draw_plots`).

```
glupredkit set_unit --use-mgdl [True|False]
```


#### Example
```
glupredkit set_unit --use-mgdl False
```

---

That's it! You can now run the desired command with the mentioned arguments. Always refer back to this guide for the correct usage.




## Contributing with code

In this section we will explain how you can contribute with enhancing the implementation of parsers, preprocessors, models, evaluation metrics and plots.

The following figure is a class diagram of the main components in GluPredKit.
<!-- ![img.png](https://miriamkw.folk.ntnu.no/figures/UML_diagram.png) -->
![img.png](https://miriamkw.folk.ntnu.no/figures/UML_diagram.png)

### Contributing With New Components

Regardless of the component type you're contributing, follow these general steps:

1. Navigate to the corresponding directory in `glupredkit/`.
2. Create a new Python file for your component.
3. Implement your component class, inheriting from the appropriate base class.
4. Add necessary tests and update the documentation.

Here are specifics for various component types:

#### Parsers
Refers to the fetching of data from data sources (for example Nighscout, Tidepool or Apple Health), and to process the data into the same table. 
- Directory: `glupredkit/parsers`
- Base Class: `BaseParser`

All the parsers should give an output of the same format. Some essential details are:
- We use pandas DataFrames as output.
- All datatypes are resampled into 5-minute intervals. If the sample rate is more frequent, the data should be aggregated with for example sum or mean values. If less frequent, NaN values should be used. Interpolation is handled in the preprocessor.
- The index-column of the dataframe should be the datetime of the sample.
- Blood glucose values should have the column name "CGM".
- Carbohydrate intake values should have the column name "carbs".
- Insulin infusion values should have the column name "insulin". 
- Additional columns and column names are optional.

#### Preprocessors
Refers to the preprocessing of the raw datasets from the parsing-stage. This includes imputation, feature transformation, splitting data etc.
- Directory: `glupredkit/preprocessors`
- Base Class: `BasePreprocessor`

Note that time-lagged features and other library-specific data-processing is handled in the model implementations, in the method `process_data` in `base_model.py`. 
That is because different libraries like scikit-learn, Keras or PyTorch, or model approaches might expect different data-input formats.
For example, time-lagged features might be stored in different ways as in separate columns or as lists in a single column.

#### Machine Learning Prediction Models
Refers to using preprocessed data to train a blood glucose prediction model.
- Directory: `glupredkit/models`
- Base Class: `BaseModel`

The method `process_data` in `base_model.py` handles addition of time-lagged features, removal of NaN values and other 
library- or model-specific configurations for the model data input format. 

#### Evaluation Metrics
Refers to different 'scores' to describing the accuracy of the predictions of a blood glucose prediction model.    
- Directory: `glupredkit/metrics`
- Base Class: `BaseMetric`

#### Evaluation Plots
Different types of plots that can illustrate blood glucose predictions together with actual measured values.
- Directory: `glupredkit/plots`
- Base Class: `BasePlot`

Remember to adhere to our coding and documentation standards when contributing!


### Testing
To run the tests, write `python tests/test_all.py` in the terminal.


## Disclaimers and limitations
* Datetimes that are fetched from Tidepool API are received converted to timezone offset +00:00. There is no way to get information about the original timezone offset from this data source.
* Bolus doses that are fetched from Tidepool API does not include the end date of the dose delivery.
* Metrics assumes mg/dL for the input.
* Note that the difference between how basal rates are registered. Bolus doses are however consistent across. Hopefully it is negligible.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details


