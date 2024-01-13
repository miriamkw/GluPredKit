import itertools
import time
from sklearn.base import BaseEstimator
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
from glupredkit.helpers.scikit_learn import process_data
from .base_model import BaseModel

from keras.layers import Dense,LSTM, Bidirectional
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        self.ph = prediction_horizon // 5 # 5 minute intervals, distance_target
        self.lookback = 15
        self.model = None

    
    def fit(self, X_train, Y_train):
        results = self._feature_selection(X_train, Y_train)
        print("results:", results)
        X_train = self._sequence(X_train)
        self.model = self._build_model(X_train.shape[1:])
        validation_size = 0.2
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, shuffle=False)
        self.model.fit(X_train, Y_train, validation_data=(X_val,Y_val), epochs=100, batch_size=128, shuffle=False, verbose=True)
        return self

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
        print("X_train[to_use]: {}".format(X_train[list(to_use)]))
        X_train = X_train[list(to_use)]
        X_train = self._sequence(X_train)
        self.model = self._build_model(X_train.shape[1:])
        validation_size = 1 - split_prc
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, shuffle=False)
        history = self.model.fit(X_train, Y_train, validation_data=(X_val,Y_val), epochs=50, batch_size=128, shuffle=False, verbose=True)
        val_loss = history.history['val_loss']
        print("val_loss: {}".format(val_loss))
        return val_loss

    def predict(self, X_test):
        X_test_seq = self._sequence(X_test)  # Renamed variable for clarity
        predictions = self.model.predict(X_test_seq)
        return predictions 
    
    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
    
    def _build_model(self, input_shape, lr=0.001, verbose = True):
        dropout = 0.1
        recurrent_dropout = 0.2

        model = Sequential()
        model.add(Bidirectional(LSTM(128,
                    return_sequences = True,
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    stateful = False),
                    input_shape = input_shape))
        model.add(LSTM(64, return_sequences = True,
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    stateful = False))
        model.add(LSTM(32,
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    stateful = False))
        model.add(Dense(1, activation='linear'))

        optimizer = RMSprop(lr = lr)
        model.compile(optimizer=optimizer, loss='mse')
        
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
    
    def _sequence(self, data):
        print("data.shape = ", data.shape)
        pad_row = data.iloc[-1].to_numpy()
        for _ in range(self.lookback):
            data = pd.concat([data, pd.DataFrame([pad_row], columns=data.columns)], ignore_index=True)
            #data = pd.concat(data, data.iloc[-1])
        print("data.shape = ", data.shape)
        sequences = []
        for i in range(len(data)):
            end_ix = i + self.lookback
            if end_ix >= len(data):
                break
            seq = data.iloc[i:end_ix, :].to_numpy()
            sequences.append(seq)
        return np.array(sequences)