import os

ohio_ids_2018 = ["559", "563", "570", "575", "588", "591"]
ohio_ids_2020 = ["540", "544", "552", "567", "584", "596"]
model_names = ""
"""
for subject in ohio_ids_2018:
    model_names = model_names + f"ridge__{subject}__30.pkl,"
    model_names = model_names + f"ridge__{subject}__60.pkl,"
    model_names = model_names + f"lstm__{subject}__30.pkl,"
    model_names = model_names + f"lstm__{subject}__60.pkl,"
    #os.system(f"python -m glupredkit.cli train_model ridge {subject}")
    #os.system(f"python -m glupredkit.cli train_model lstm {subject}")

for subject in ohio_ids_2020:
    model_names = model_names + f"ridge__{subject}__30.pkl,"
    model_names = model_names + f"ridge__{subject}__60.pkl,"
    model_names = model_names + f"lstm__{subject}__30.pkl,"
    model_names = model_names + f"lstm__{subject}__60.pkl,"
    #os.system(f"python -m glupredkit.cli train_model ridge {subject}")
    #os.system(f"python -m glupredkit.cli train_model lstm {subject}")
"""
model_names = model_names + "ridge__me__30.pkl,ridge__me__60.pkl,lstm__me__30.pkl,lstm__me__60.pkl"

os.system(f"python -m glupredkit.cli calculate_metrics --metrics clarke_error_grid --models {model_names}")

"""
Script to convert a Ridge Regressor and LSTM (Tensorflow) to CoreML.
"""
"""
import coremltools as ct
import pickle
import tensorflow as tf

horizons = ['15', '30', '45', '60', '75', '90', '105', '105', '120', '135', '150', '165', '180']
# models = ['ridge', 'lstm']
models = ['lstm']
config = 'me'

for horizon in horizons:
    for model in models:
        file_name = f'{model}__{config}__{horizon}'
        file_path = f"data/trained_models/{file_name}.pkl"

        # Load model class
        with open(file_path, 'rb') as file:
            loaded_class = pickle.load(file)

        if model == 'ridge':
            model = loaded_class.model.best_estimator_.named_steps['regressor']
            feature_names = loaded_class.model.best_estimator_.feature_names_in_
            mlmodel = ct.converters.sklearn.convert(model,
                                                    feature_names)

        if model == 'lstm':
            model = tf.keras.models.load_model(loaded_class.model_path,
                                               custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
            mlmodel = ct.convert(model,
                                 # source='tensorflow',
                                 )

        output_mlmodel_file = f'{file_name}.mlpackage'
        mlmodel.save(f"data/trained_models/{output_mlmodel_file}")

"""
