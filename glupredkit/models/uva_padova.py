from glupredkit.models.base_model import BaseModel
from py_replay_bg.py_replay_bg import ReplayBG
from py_replay_bg.identification.mcmc import MCMC
from py_replay_bg.model.t1d_model import T1DModel
from py_replay_bg.data import ReplayBGData
from filterpy.monte_carlo import systematic_resample
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import systematic_resample

from filterpy.kalman import unscented_transform
from scipy.stats import norm
from copy import copy
from pandas import DataFrame, Timedelta, TimedeltaIndex
import numpy as np
import pickle
import pandas as pd
import os
import shutil


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        # TODO: Should be an empty list, one for each subject id
        self.rbg = None

    def fit(self, x_train, y_train):
        x_train = self.process_input_data(x_train)

        # Fit parameters of ReplayBG object
        modality = 'identification'  # set modality as 'identification'
        bw = 80  # Placeholder body weight
        scenario = 'single-meal'
        cgm_model = 'CGM'
        n_steps = 1000  # set the number of steps that will be used for identification (for multi-meal it should be at least 100k)

        # TODO: Remove this?
        subset_df = x_train[:12 * 24]

        # TODO: For each subject

        # TODO: Can we turn off saving and plotting results?
        # print(subset_df.info())
        # subset_df.to_csv('test_data.csv')

        self.rbg = ReplayBG(modality=modality, data=subset_df, bw=bw, scenario=scenario, save_name='', save_folder='',
                            n_steps=n_steps, cgm_model=cgm_model, plot_mode=False, analyze_results=True)
        # Run identification
        self.rbg.run(data=subset_df, bw=bw)
        """

        # rbg_data = ReplayBGData(data=subset_df, rbg=rbg)
        # self.draws = rbg.mcmc.identify(data=x_train, rbg_data=rbg_data, rbg=rbg)

        # TODO: Get draws from the stored file
        with open(os.path.join(self.rbg.environment.replay_bg_path, 'results', 'draws',
                               'draws_' + self.rbg.environment.save_name + '.pkl'), 'rb') as file:
            identification_results = pickle.load(file)
        draws = identification_results['draws']
        """

        # Delete the stored draws after run
        shutil.rmtree('results')

        return self

    def predict(self, x_test):
        """
        Model parameters:
        - bw: Body weight
        - Beta: The delay of the carbohydrate effect
        - u2ss: Basal rate in mU/(kg*min) = 0,2 / U / hr if bw = 80.
        """
        x_test = self.process_input_data(x_test)
        mp = self.rbg.model.model_parameters
        # From ReplayBG (the code there is commented out):
        mp.ka1 = 0.0034  # 1/min (virtually 0 in 77% of the cases)

        print("bw:", mp.bw)
        print("beta", mp.beta)
        print("tau:", mp.tau)
        print("u2ss:", mp.u2ss)
        print("ka1:", mp.ka1)
        print("kd:", mp.kd)
        print("ka2:", mp.ka2)
        print("Xpb:", mp.Xpb)

        # TODO: Only for testing, remove when finished
        subset_x_test = x_test[:10]

        # TODO: Iterate for each id in the data
        prediction_result = self.get_phy_prediction(mp, subset_x_test, 30)

        print("predictions: ", prediction_result)

        return

    def get_phy_prediction(self, model_parameters, data, prediction_horizon):
        """
        This function is translated from MatLab: https://github.com/checoisback/phy-predict/blob/main/getPhyPrediction.m.

        Computes the glucose predicted profiles by using a non-linear physiological model of glucose-insulin dynamics
        integrated into a particle filter.

        Inputs:
        - identified_model_parameters: a dictionary containing the identified model parameters described in the provided link.
        - test_data: a DataFrame containing the following columns: Time, glucose, CHO, bolus_insulin, and basal_insulin.
        The measurement units must be consistent with the ones described in the provided link.
        - PH: the prediction horizon (in minutes).

        Output:
        - prediction_results: a dictionary containing the test data (prediction_results['data']) and the predicted profile
        (prediction_results['ighat']).
        """
        Ts = 5  # minutes
        n_particles = 5000  # number of particles
        sigma_v = 25  # measurements noise variance
        sigma_u0 = np.array([10, 10, 10, 0.6, 0.6, 0.6, 1e-6, 10, 10])  # process noise variance

        # set physiological model environment
        model = {
            'TS': 1,  # physiological model sampling time [minute]
            'YTS': 5,  # raw data sampling time [minute]
            'TID': (data['t'].iloc[-1] - data['t'].iloc[0]).total_seconds() / 60 + 1,  # [min] from 1 to TID identify
            # the model parameters [min]
        }
        model['TIDSTEPS'] = int(model['TID'] / model['TS'])  # integration steps
        # model['TIDYSTEPS'] = model['TID'] / model['YTS']  # total identification simulation time [sample steps]
        model['TIDYSTEPS'] = data.shape[0]  # total identification simulation time [sample steps]
        model_parameters.TS = model['TS']

        # prepare input data
        bolus, basal, bolusDelayed, basalDelayed = self.insulin_setup_pf(data, model, model_parameters)
        cho, choDelayed = self.meal_setup_pf(data, model, model_parameters)
        meal = choDelayed
        total_ins = bolusDelayed + basalDelayed

        end_time = data['t'].iloc[0] + pd.Timedelta(minutes=len(cho) - 1)
        time_data = pd.date_range(start=data['t'].iloc[0], end=end_time, freq='min')

        # initial state
        x0 = np.array([
            0,  # Qsto1(0)
            0,  # Qsto2(0)
            0,  # Qgut(0)
            model_parameters.u2ss / (model_parameters.ka1 + model_parameters.kd),  # Isc1(0)
            (model_parameters.kd / model_parameters.ka2) * model_parameters.u2ss / (
                        model_parameters.ka1 + model_parameters.kd),  # Isc2(0)
            (model_parameters.ka1 / model_parameters.ke) * model_parameters.u2ss / (
                        model_parameters.ka1 + model_parameters.kd) +
            (model_parameters.ka2 / model_parameters.ke) * (model_parameters.kd / model_parameters.ka2) *
            model_parameters.u2ss / (model_parameters.ka1 + model_parameters.kd),  # Ip(0)
            model_parameters.Xpb,  # X(0)
            data['glucose'].iloc[0],  # G(0)
            data['glucose'].iloc[0]  # IG(0)
        ])

        sigma0 = np.eye(len(x0))

        # Set state bounds
        state_bound = np.ones((len(x0), 2))
        state_bound[:, 0] = np.array(x0) - 0.03 * np.array(x0)  # Lower bound
        state_bound[:, 1] = np.array(x0) + 0.03 * np.array(x0)  # Upper bound

        # Initialize the particle filter
        pf_v0 = ParticleFilter(gi_particle_filter_state_function, gi_measurement_likelihood_function, n_particles, x0, np.diag(sigma0))

        last_best_guess, last_best_cov, G_hat, IG_hat, VarG_hat, VarIG_hat = self.apply_pf(pf_v0, time_data,
                                                                                      data['glucose'], meal,
                                                                                      total_ins, x0, sigma0, sigma_u0,
                                                                                      sigma_v, model_parameters,
                                                                                      prediction_horizon)

        predicted_CGM = IG_hat[:-int(prediction_horizon / Ts)]

        ighat = DataFrame(data=predicted_CGM, index=time_data[int(prediction_horizon / Ts):], columns=['glucose'])

        prediction_results = {'dataTest': data, 'dataHat': ighat}
        return prediction_results

    def insulin_setup_pf(self, data, model, model_parameters):
        """
        Generates the vectors containing the insulin infusions to be used to simulate the physiological model.

        Inputs:
            - data: a DataFrame containing the data to be used by the tool;
            - model: a dictionary that contains general parameters of the physiological model;
            - model_parameters: a dictionary containing the model parameters.
        Outputs:
            - bolus: a vector containing the insulin bolus dose at each time step [mU/min*kg];
            - basal: a vector containing the basal insulin value at each time step [mU/min*kg];
            - bolusDelayed: a vector containing the insulin bolus dose at each time step delayed by tau min [mU/min*kg];
            - basalDelayed: a vector containing the basal insulin value at each time step delayed by tau min [mU/min*kg].
        """
        # Initialize the basal and bolus vectors
        basal = np.zeros(int(model['TIDSTEPS']))
        bolus = np.zeros(int(model['TIDSTEPS']))

        # Set the basal and bolus vectors
        for time in range(len(np.arange(0, model['TID'], model['YTS']))):
            start_index = int((1 + time - 1) * (model['YTS'] / model['TS']))
            end_index = int(time * (model['YTS'] / model['TS']) + 1)

            bolus[start_index:end_index] = data['bolus'].iloc[time] * 1000 / model_parameters.bw  # mU/(kg*min)
            basal[start_index:end_index] = data['basal'].iloc[time] * 1000 / model_parameters.bw  # mU/(kg*min)

        # Add delay in insulin absorption
        bolus_delay = int(np.floor(model_parameters.tau / model['TS']))
        bolus_delay = int(np.round(bolus_delay / 5) * 5)
        bolus_delayed = np.concatenate((np.zeros(bolus_delay), bolus))
        bolus_delayed = bolus_delayed[:int(model['TIDSTEPS'])]

        basal_delayed = np.concatenate((np.ones(bolus_delay) * basal[0], basal))
        basal_delayed = basal_delayed[:int(model['TIDSTEPS'])]

        basal = basal[:int(model['TIDSTEPS'])]
        bolus = bolus[:int(model['TIDSTEPS'])]

        return bolus, basal, bolus_delayed, basal_delayed

    def meal_setup_pf(self, data, model, model_parameters):
        """
        Generates the vector containing the CHO intake events to be used to simulate the physiological model.

        Inputs:
            - data: a DataFrame containing the data to be used by the tool;
            - model: a dictionary that contains general parameters of the physiological model;
            - model_parameters: a dictionary containing the model parameters;
            - environment: a dictionary that contains general parameters to be used by ReplayBG.
        Outputs:
            - meal: a vector containing the carbohydrate intake at each time step [mg/min*kg];
            - mealDelayed: a vector containing the carbohydrate intake at each time step delayed by beta min [mg/min*kg].
        """
        # Initialize the meal vector
        meal = np.zeros(model['TIDSTEPS'])

        # Set the meal vector
        for time in range(len(np.arange(0, model['TID'], model['YTS']))):
            start_index = int((1 + time - 1) * (model['YTS'] / model['TS']))
            end_index = int(time * (model['YTS'] / model['TS']) + 1)
            meal[start_index:end_index] = data['cho'].iloc[time] * 1000 / model_parameters.bw  # mg/(kg*min)

        # Add delay of main meal absorption
        meal_delay = int(round(model_parameters.beta / model['TS']))
        meal_delay = int(round(meal_delay / 5) * 5)
        meal_delayed = np.concatenate((np.zeros(meal_delay), meal))
        meal_delayed = meal_delayed[:model['TIDSTEPS']]
        meal = meal[:model['TIDSTEPS']]

        return meal, meal_delayed

    def apply_pf(self, pf, time, noisy_measure, meal, total_ins, x0, sigma0, sigma_u0, sigma_v, model_parameters,
                                                                                      prediction_horizon):
        """
        Applies a particle filter to perform real-time filtering.

        Inputs:
            - pf: a particle filter object
            - time: array containing the time information
            - noisy_measure: array containing the noisy measurements
            - meal: array containing the carbohydrate intake at each time step [mg/min*kg]
            - total_ins: array containing the total insulin intake at each time step [mU/min*kg]
            - kwargs: additional keyword arguments including:
                - InitialCondition_x0: initial state condition
                - InitialCondition_sigma0: initial state covariance
                - processNoiseVariance: process noise variance
                - measurementNoiseVariance: measurement noise variance
                - parameterStructure: structure containing model parameters
                - predictionHorizon: prediction horizon (default: 30)

        Outputs:
            - lastBestGuess: matrix containing the last best guess of state variables
            - lastBestCov: matrix containing the last best covariance of state variables
            - G_hat: matrix containing the predicted glucose values
            - IG_hat: matrix containing the predicted interstitial glucose values
            - VarG_hat: matrix containing the predicted variance of glucose
            - VarIG_hat: matrix containing the predicted variance of interstitial glucose
        """

        sigma0 = sigma0
        sigma_u = np.sqrt(sigma_u0)
        sigma_v = np.sqrt(sigma_v)

        lastBestGuess = np.zeros((len(x0), len(noisy_measure)))
        lastBestGuess[7, 0] = noisy_measure[0]  # Initial glucose
        lastBestGuess[8, 0] = noisy_measure[0]  # Initial interstitial glucose
        lastBestCov = np.zeros((len(x0), len(noisy_measure)))

        IG_hat = np.zeros((len(noisy_measure), prediction_horizon))
        G_hat = np.zeros((len(noisy_measure), prediction_horizon))
        VarIG_hat = np.zeros((len(noisy_measure), prediction_horizon))
        VarG_hat = np.zeros((len(noisy_measure), prediction_horizon))

        state_corrected = np.zeros(len(x0))
        cov_corrected = np.zeros((len(x0), len(x0)))

        for k in range(len(time)):
            # Prediction step
            state_pred, cov_pred = pf.predict(meal[k], total_ins[k], time[k].hour * 3600 + time[k].minute * 60,
                                              sigma_u, model_parameters)

            index_measure = k if (k - 1) % (5 / model_parameters.TS) == 0 else None
            CGM_measure = noisy_measure[index_measure] if index_measure is not None else np.nan

            # Correction step
            if not np.isnan(CGM_measure):
                state_corrected, cov_corrected = pf.correct(CGM_measure, sigma_v)
            else:
                state_corrected = state_pred
                cov_corrected = cov_pred

            if index_measure is not None:
                lastBestGuess[:, index_measure] = state_corrected
                lastBestCov[:, index_measure] = np.diag(cov_corrected)

            # k-step ahead prediction
            if (k + prediction_horizon < len(time)) and (index_measure is not None):
                pf_pred = copy(pf)
                mP_pred = copy(model_parameters)

                for p in range(prediction_horizon):
                    step_ahead_prediction, cov_ahead_prediction = pf_pred.predict(meal[k + p], total_ins[k + p],
                                                                                  time[k + p].hour * 3600 +
                                                                                  time[k + p].minute * 60, sigma_u,
                                                                                  mP_pred)
                    G_hat[index_measure, p] = step_ahead_prediction[7]  # Glucose prediction
                    IG_hat[index_measure, p] = step_ahead_prediction[8]  # Interstitial glucose prediction

                    pred_variance = np.diag(cov_ahead_prediction)
                    VarG_hat[index_measure, p] = pred_variance[7]  # Variance of glucose prediction
                    VarIG_hat[index_measure, p] = pred_variance[8]  # Variance of interstitial glucose prediction

        return lastBestGuess, lastBestCov, G_hat, IG_hat, VarG_hat, VarIG_hat

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        # You need to fill this method with your logic
        # For example:
        # return self.model.best_params_
        return

    def process_input_data(self, x_df):
        x_df = x_df.copy()
        x_df.rename(columns={'CGM': 'glucose', 'carbs': 'cho'}, inplace=True)
        x_df['SMBG'] = np.nan
        x_df['CR'] = np.nan
        x_df['CF'] = np.nan
        x_df['basal'] = x_df['basal'] / 60  # Basal needs to be in U and not U/hr
        x_df['cho_label'] = ''
        x_df['bolus_label'] = ''
        x_df['exercise'] = 0
        x_df['t'] = x_df.index
        x_df.reset_index(inplace=True)

        return x_df

    def process_data(self, df, model_config_manager, real_time):
        df = df.dropna()
        return df


def gi_measurement_likelihood_function(particle, measurement, sigma_v):
    # The measurement is the ninth state. Extract all measurement hypotheses from particles
    #predictedMeasurement = particles[8, :]
    predicted_measurement = particle[0]

    # Calculate the likelihood of each predicted measurement
    likelihood = norm.pdf(predicted_measurement, measurement, sigma_v)

    return likelihood


def gi_particle_filter_state_function(particles, CHO, INS, time, sigma_u, mP):
    numberOfParticles, numberOfStates = particles.shape

    # Time-propagate each particle using Euler integration
    dt = mP.TS  # Sample time
    for kk in range(numberOfParticles):
        state_change = gi_state_function_continuous(particles[kk, :], CHO, INS, time, mP) * dt
        particles[kk, :] += state_change

    # Add Gaussian noise with specified variance processNoise
    process_noise = np.diag(sigma_u * np.ones(numberOfStates))  # Create diagonal matrix from sigma_u
    noise = np.dot(process_noise, np.random.randn(numberOfStates, numberOfParticles)).T  # Transpose to fit particles
    particles += noise

    return particles


def gi_state_function_continuous(x, CHO, INS, time, mP):
    dxdt = np.zeros_like(x)
    Qsto1, Qsto2, Qgut, Isc1, Isc2, Ip, X, G, IG = x

    # Calculate state derivatives
    SI = mP.SI
    risk = compute_hypoglycemic_risk(x[7], mP)  # Assuming G is x[7]
    rhoRisk = 1 + risk

    Ra = mP.f * mP.kabs * Qgut

    dxdt[0] = -mP.kgri * Qsto1 + CHO
    dxdt[1] = mP.kgri * Qsto1 - mP.kempt * Qsto2
    dxdt[2] = mP.kempt * Qsto2 - mP.kabs * Qgut
    dxdt[3] = -mP.kd * Isc1 + INS
    dxdt[4] = mP.kd * Isc1 - mP.ka2 * Isc2
    dxdt[5] = mP.ka2 * Isc2 - mP.ke * Ip
    dxdt[6] = -mP.p2 * (X - (SI / mP.VI) * (Ip - mP.Ipb))
    dxdt[7] = -((mP.SG + rhoRisk * X) * G) + mP.SG * mP.Gb + Ra / mP.VG
    dxdt[8] = -(1 / mP.alpha) * (IG - G)
    return dxdt


def compute_hypoglycemic_risk(G, mP):
    # Function to compute hypoglycemic risk as described in Visentin et al., JDST, 2018.

    # Setting the risk model threshold
    Gth = 60

    # Compute the risk
    Gb = mP.Gb
    risk = (
            10 * (np.log(G) ** mP.r2 - np.log(Gb) ** mP.r2) ** 2 * ((G < Gb) & (G >= Gth)) +
            10 * (np.log(Gth) ** mP.r2 - np.log(Gb) ** mP.r2) ** 2 * (G < Gth)
    )
    risk = abs(risk)

    return risk


class ParticleFilter:
    def __init__(self, state_transition_fn, measurement_fn, num_particles, x0, sigma0):
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, len(x0)) * np.sqrt(sigma0) + x0
        self.weights = np.ones(num_particles) / num_particles
        self.state_transition_fn = state_transition_fn
        self.measurement_fn = measurement_fn

    def predict(self, carbs, insulin, time, sigma_u, model_parameters):
        """Predict the next state of the particles."""
        self.particles = self.state_transition_fn(self.particles, carbs, insulin, time, sigma_u, model_parameters)

        # Calculate the mean of the particles
        mean_state = np.mean(self.particles, axis=0)

        # Calculate the covariance matrix of the particles
        cov_state = np.cov(self.particles, rowvar=False)  # rowvar=False to treat rows as variables

        return mean_state, cov_state

    def update(self, measurement, sigma_v):
        """Update the particle weights based on measurement likelihood."""
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights *= self.measurement_fn(distances, sigma_v)
        self.weights += 1.e-300      # avoid divide by zero
        self.weights /= np.sum(self.weights)


    def correct(self, measurement, sigma_v):
        """
        Correct/update the particle weights based on the measurement, and resample.

        Parameters:
            measurement (float): The new measurement.
            sigma_v (float): Measurement noise standard deviation.

        Returns:
            tuple: Corrected state estimate and its covariance.
        """
        # Update weights based on measurement likelihood
        for i in range(len(self.particles)):
            self.weights[i] *= self.measurement_fn(self.particles[i], measurement, sigma_v)

        # Normalize weights
        self.weights += 1e-300  # avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample particles based on updated weights
        indices = self.resample()
        self.particles = self.particles[indices]
        self.weights = np.ones(len(self.particles)) / len(self.particles)  # Reset weights

        # Calculate corrected state estimate and covariance
        corrected_state = np.mean(self.particles, axis=0)
        corrected_cov = np.cov(self.particles, rowvar=False)

        return corrected_state, corrected_cov

    def resample(self):
        """
        Resample particles based on the weights.

        Returns:
            array: Indices of particles after resampling.
        """
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # ensure sum is exactly 1.0
        indices = np.searchsorted(cumulative_sum, np.random.random(len(self.weights)))
        return indices

    def estimate(self):
        """Estimate the current state as the mean of the particles."""
        return np.average(self.particles, weights=self.weights, axis=0)

