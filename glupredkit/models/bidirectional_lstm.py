from datetime import datetime
import itertools
import time
from sklearn.base import BaseEstimator
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
# from glupredkit.helpers.scikit_learn import process_data
from glupredkit.helpers.tf_keras import process_data
from .base_model import BaseModel
import tensorflow as tf
from keras.layers import Dense,LSTM, Bidirectional, Dropout
from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import ast

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        # self.lookback = 15
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"
    
    def fit(self, x_train, y_train):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_train['sequence']]
        targets = y_train.tolist()
        x_train = np.array(sequences)
        y_train = np.array(targets)
        print("x_train.shape: ", x_train.shape)

        model = self._build_model(x_train.shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss', mode='min')
        model.fit(x_train, y_train, epochs=100, shuffle=False, verbose=True, callbacks=[early_stopping, reduce_lr], validation_split=0.2)
        model.save(self.model_path)
        return self

    def predict(self, x_test):
        sequences = [np.array(ast.literal_eval(seq_str)) for seq_str in x_test['sequence']]
        x_test = np.array(sequences)
        model = tf.keras.models.load_model(self.model_path, custom_objects={"Adam": tf.keras.optimizers.legacy.Adam})
        predictions = model.predict(x_test)
        return [val[0] for val in predictions]
    
    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        model_config_manager.config["num_lagged_features"] = 3
        # model_config_manager.config["num_features"] = ["CGM", "carbs", "bolus"]
        return process_data(df, model_config_manager, real_time)
    
    def _build_model(self, input_shape, lr=0.001, verbose = True):
        dropout = 0.1
        recurrent_dropout = 0.2

        model = Sequential()
        '''
        model.add(Bidirectional(LSTM(128,
                    return_sequences = True,
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    stateful = False),
                    input_shape = (input_shape[1], input_shape[2])))
        model.add(LSTM(64, return_sequences = True,
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    stateful = False))
        model.add(LSTM(32,
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    stateful = False))
        model.add(Dense(1, activation='linear'))
        '''
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(0.001)), input_shape=(input_shape[1], input_shape[2])))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        # optimizer = RMSprop(lr = lr)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        
        if verbose:
            model.summary()

        return model
    
    def _split_sequences(self, X_data, Y_data, n_outputs=1):
        # Adjust input and output data
        sequences = np.concatenate((X_data, np.array(Y_data).reshape(-1, 1)), axis=1)
        
        # Prepare data for LSTM
        X, y = list(), list()

        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.lookback
            # check if we are beyond the dataset
            if (end_ix + n_outputs - 1) > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[(end_ix - 1):(end_ix - 1 + n_outputs), -1]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)
    
    # def _sequence(self, data):
    #     # print("data.shape = ", data.shape)
    #     # res = data[['CGM','carbs']].shift(-self.lookback)
    #     # print(res)

    #     pad_row = data.iloc[-1].to_numpy()
    #     for _ in range(self.lookback):
    #         data = pd.concat([data, pd.DataFrame([pad_row], columns=data.columns)], ignore_index=True)
    #         #data = pd.concat(data, data.iloc[-1])
    #     print("data.shape = ", data.shape)
    #     sequences = []
    #     for i in range(len(data)):
    #         end_ix = i + self.lookback
    #         if end_ix >= len(data):
    #             break
    #         seq = data.iloc[i:end_ix, :].to_numpy()
    #         sequences.append(seq)
    #     return np.array(sequences)
        '''
    def _feature_selection(self, X_train, Y_train):
        feat_set = ['carbs'] # , 'bolus', 'basal' candidates for selection
        power_feat_set = list(itertools.chain.from_iterable(itertools.combinations(feat_set, r) for r in range(1, len(feat_set) + 1)))
        for n, el in enumerate(power_feat_set):
            power_feat_set[n] = ('CGM',) + el

        hyperparameter = dict()
        results = dict()
        results['val_loss'] = list()
        results['to_use'] = list()
        results['split_prc'] = list()
        hyperparameter['to_use'] = list()
        for f in power_feat_set:
            hyperparameter['to_use'].append(f)
        hyperparameter['to_use'].append(('CGM',))
        hyperparameter['split_prc'] = [0.5, 0.6, 0.7, 0.8]
        iteration = 1
        n_iteration = len(hyperparameter['to_use']) * len(hyperparameter['split_prc'])
        n_rep = 3

        lr = 0.001
        print("hyperparameter:", hyperparameter)
        start = time.time()
        for split_prc in hyperparameter['split_prc']:
            for to_use in hyperparameter['to_use']:
                val_loss_list = list()
                print(F"Iteration: {iteration}/{n_iteration}, Feats: {'-'.join(to_use)}, SPLT: {split_prc}")

                for _ in np.arange(n_rep):
                    val_loss = self._fit_model(X_train, Y_train, to_use, split_prc)
                    val_loss_list.append(val_loss)

                results['val_loss'].append(np.mean(val_loss_list))				
                results['to_use'].append(to_use)
                results['split_prc'].append(split_prc)

                iteration += 1
                end = time.time()
                print(F"Elapsed: {(end-start)/60/60:.2f} hours")
        return results

    def _fit_model(self, X_train, Y_train, to_use, split_prc):
        self.model = self._build_model(X_train.shape[1:])
        validation_size = 1 - split_prc
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, shuffle=False)
        history = self.model.fit(X_train, Y_train, validation_data=(X_val,Y_val), epochs=50, batch_size=128, shuffle=False, verbose=True)
        val_loss = history.history['val_loss']
        print("val_loss: {}".format(val_loss))
        return val_loss
    '''