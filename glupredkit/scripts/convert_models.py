import coremltools as ct
import pickle
import tensorflow as tf

horizons = ['180']
models = ['ridge_multioutput_constrained']
config = 'me_multioutput'

for horizon in horizons:
    for model in models:
        file_name = f'{model}__{config}__{horizon}'
        file_path = f"data/trained_models/{file_name}.pkl"

        # Load model class
        with open(file_path, 'rb') as file:
            loaded_class = pickle.load(file)

        # Code for converting models trained using Scikit Learn
        if model == 'ridge':
            model = loaded_class.model.best_estimator_
            feature_names = model.feature_names_in_
            mlmodel = ct.converters.sklearn.convert(model,
                                                    feature_names)

        if model == 'ridge_multioutput':
            loaded_class.save_model_weights(file_path=f'data/trained_models/ridge_multioutput__{config}__{horizon}.json')
            break

        if model == 'ridge_multioutput_constrained':
            loaded_class.print_coefficients()
            loaded_class.save_model_weights(file_path=f'data/trained_models/ridge_multioutput_constrained__{config}__{horizon}__amplified.json')
            break

        # Code for converting models trained using Tensorflow or PyTorch
        if model == 'lstm':
            model = tf.keras.models.load_model(loaded_class.model_path,
                                               custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
            mlmodel = ct.convert(model,
                                 source='tensorflow',
                                 # source='pytorch',
                                 )

        output_mlmodel_file = f'{file_name}.mlpackage'
        mlmodel.save(f"data/trained_models/{output_mlmodel_file}")
