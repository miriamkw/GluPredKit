import numpy as np
from pyloopkit.exponential_insulin_model import percent_effect_remaining


def get_average_glucose_penalty_for_pairs(y_true, y_pred, penalty_function):
    """
	Get the glucose penalty for a collection of pairs of measured and predicted values, given a specific penalty function.
	Method documented here: https://docs.google.com/document/d/14AJ9u2oGJiiJU1cWVDf_rC_WdJc0oOj1uIkXutOovQU/edit#

	Parameters:
	y_true -- list of measured values
	y_pred -- list of predicted values
	penalty_function -- a function that takes a blood glucose value as input and returns the penalty

	Output:
	Average penalty over all pairs
	"""
    penalties = [get_glucose_penalty_for_pair(true, pred, penalty_function) for true, pred in zip(y_true, y_pred)]
    return np.mean(penalties)


def get_glucose_penalty_for_pair(true_value, pred_value, penalty_function, target=112.5):
    """
	Get the glucose penalty for one pair of measured and predicted values, given a specific penalty function.

	Parameters:
	true_value -- one measured glucose value
	pred_value -- one predicted glucose value
	penalty_function -- a function that takes a blood glucose value as input and returns the penalty

	Output:
	Penalty for one pair of measured and predicted value
	"""
    # Calculate ideal and true trajectory
    ideal_glucose_values = get_treatment_trajectory(true_value, target)
    derived_end_value = true_value + target - pred_value
    derived_glucose_values = get_treatment_trajectory(true_value, derived_end_value)

    # Calculate average penalty for the trajectories
    penalty_ideal_treatment = np.mean([penalty_function(value) for value in ideal_glucose_values])
    penalty_derived_treatment = np.mean([penalty_function(value) for value in derived_glucose_values])

    return penalty_derived_treatment - penalty_ideal_treatment


def get_treatment_trajectory(start_value, end_value):
    """
    Get the blood glucose trajectory for six hours for a given a start and end glucose value, assuming that
    we are aiming for a specific target.

    Arguments:
    start_value -- the blood glucose measurement at the beginning of a trajectory
    end_value -- the end blood glucose in mg/dL given a treatment action that will lead to this value

    Output:
    A list of glucose_values representing the trajectory given a start and end value
    """
    glucose_values = [start_value]

    # Hard coding the insulin model parameters for now
    action_duration = 360.0
    peak_activity_time = 75.0
    delay = 10.0
    interval = 5

    # Minutes of total trajectory
    total_time = 360
    n = int(360 / interval)
    t = 0

    # TODO: Derive explicit formulas for calculating loss for the ideal treatments

    for i in range(n):
        t = t + interval

        if start_value <= end_value:
            glucose_values.append(get_glucose_from_carbs(start_value, end_value, t))
        else:
            glucose_values.append(
                start_value - (1 - percent_effect_remaining(t - delay, action_duration, peak_activity_time)) * (
                        start_value - end_value))

    assert len(glucose_values) == n + 1, \
        "expected output shape to match"

    return glucose_values


def get_glucose_from_carbs(blood_glucose, target, time):
    """
    Get the glucose value after a given time and target assuming that fast acting carbs are administered,
    which, after a ten-minute delay, raise blood glucose by 2 mg/dL/min until the target blood glucose is reached

    Parameters:
    blood_glucose -- reference blood glucose value
    target -- target blood glucose value
    time -- time in minutes after carbs are administered

    Output:
    A glucose value at a given time and given an ideal amount of carbohydrates
    """
    new_value = blood_glucose + (time - 10) * 2

    if time <= 10:
        return blood_glucose
    elif new_value >= target:
        return target
    else:
        return min(target, new_value)
