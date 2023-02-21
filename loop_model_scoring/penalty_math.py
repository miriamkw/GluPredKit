from datetime import timedelta
from pyloopkit.exponential_insulin_model import percent_effect_remaining
import numpy as np

"""
In this document you can:
- Calculate "ideal treatments"
- Calculate the penalty for a given forecast
"""

# TODO:
# Add a flexible way of choosing different penalty functions
# Add more penalty function alternatives


def get_ideal_treatment(start_value, end_value):
	""" Get the blood glucose trajectory for six hours for an ideal treatment given a start- and end-glucose value

	Arguments:
	start_value -- the blood glucose measurement at the beginning of a trajectory
	end_value -- the end blood glucose in mg/dL given a treatment action that will lead to this value

	Output:
	A list of glucose_values representing the trajectory of an ideal treatment
	"""
	# The adjustment that will be made to the glucose value
	end_value_delta = end_value - start_value

	# TODO: Derive explicit formulas for calculating loss for the ideal treatments
	# TODO: Cache the results of this function
	# TODO: Add vectorization to this function

	return [start_value] + [x + start_value for x in get_ideal_treatment_origin(end_value_delta)]


def get_ideal_treatment_origin(end_value_delta):
	glucose_values = []
	# Decrease the interval to get more accurate penalty, but slower runtime
	interval = 5

	# Minutes of total trajectory
	total_time = 360
	n = int(total_time / interval)

	for i in range(n):
		t = (i + 1) * interval

		if end_value_delta == 0:
			glucose_values.append(0)
		elif end_value_delta > 0:
			glucose_values.append(get_glucose_from_carbs(0, end_value_delta, t))
		else:
			glucose_values.append((1 - percent_effect_remaining(t - 10.0, 360.0, 75.0)) * end_value_delta)

	assert len(glucose_values) == n,\
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
	if time <= 10:
		return blood_glucose

	new_value = blood_glucose + (time - 10)*2
	return min(target, new_value)


def get_glucose_penalties_for_pairs(true_values, derived_values, target=105, penalty_type='bayer'): 
	"""
	Get the glucose penalty for a pair of measured and predicted values

	To do: Implement support for different penalty functions

	Parameters:
	true_values -- a list of measured glucose value at t_i
	derived_values -- a list of predicted glucose value at t_i

	Output:
	Losses -- a list of values of same length as imputs that can never be less than zero
	"""
	assert len(true_values) == len(derived_values),\
		"expected input shape to match"

	penalties = []

	for i in range(0, len(true_values)):
		penalty = get_glucose_penalty_for_pair(true_values[i], derived_values[i], target=target, penalty_type=penalty_type)
		penalties.append(penalty)

	return penalties


def get_glucose_penalty_for_pair(true_blood_glucose, derived_blood_glucose, target=105, penalty_type='bayer'): 
	"""
	Get the glucose penalty for a pair of measured and predicted values

	The different types of penalty functions are:
	'kovatchev' -- Kovatchev et al. 1997
	'bayer_kovatechev' -- 
	'bayer' -- Bayer (see documentation)
	'cao' -- Cao et al. 2018
	'van_herpe' - Van Herpe et al. 2008
	'error' -- Predicted minus true value, can be used to calculate rmse, mae or me

	Parameters:
	true_blood_glucose -- a measured glucose value at t_i
	derived_blood_glucose -- a predicted glucose value at t_i

	Output:
	Loss -- a value that can never be less than zero
	"""
	ideal_treatment_true = get_ideal_treatment(true_blood_glucose, target)

	derived_end_value = true_blood_glucose + target - derived_blood_glucose
	ideal_treatment_derived = get_ideal_treatment(true_blood_glucose, derived_end_value)

	if penalty_type == 'kovatchev':
		penalty_true = np.mean([get_glucose_penalty_kovatchev(x) for x in ideal_treatment_true])
		penalty_derived = np.mean([get_glucose_penalty_kovatchev(x) for x in ideal_treatment_derived])
		return penalty_derived - penalty_true
	elif penalty_type == 'bayer_kovatechev':
		penalty_true = np.mean([get_glucose_penalty_bayer_kovatchev(x) for x in ideal_treatment_true])
		penalty_derived = np.mean([get_glucose_penalty_bayer_kovatchev(x) for x in ideal_treatment_derived])
		return penalty_derived - penalty_true
	elif penalty_type == 'bayer':
		penalty_true = np.mean([get_glucose_penalty_bayer(x) for x in ideal_treatment_true])
		penalty_derived = np.mean([get_glucose_penalty_bayer(x) for x in ideal_treatment_derived])
		return penalty_derived - penalty_true
	elif penalty_type == 'cao':
		penalty_true = np.mean([get_glucose_penalty_cao(x) for x in ideal_treatment_true])
		penalty_derived = np.mean([get_glucose_penalty_cao(x) for x in ideal_treatment_derived])
		return penalty_derived - penalty_true
	elif penalty_type == 'van_herpe':
		penalty_true = np.mean([get_glucose_penalty_van_herpe(x) for x in ideal_treatment_true])
		penalty_derived = np.mean([get_glucose_penalty_van_herpe(x) for x in ideal_treatment_derived])
		return penalty_derived - penalty_true
	elif penalty_type == 'error':
		return derived_blood_glucose
	else:
		assert('Type not recognized')


def get_glucose_penalties(ideal_treatment):
	return [get_glucose_penalty_bayer(x) for x in ideal_treatment]


"""
This section consists of the different penalty approaches
"""
def get_glucose_penalty_kovatchev(blood_glucose): 
	"""
	Get the glucose penalty for a glucose value using the Bayer (105) method

	Parameters:
	blood_glucose -- one measurement / prediction

	Output:
	Penalty
	"""
	blood_glucose = max(blood_glucose, 1)
	return 10 * (1.509 * (np.log(blood_glucose)**1.084 - 5.381))**2


def get_glucose_penalty_bayer_kovatchev(blood_glucose, target=105): 
	"""
	Get the glucose penalty for a glucose value using the Bayer (105) method

	Parameters:
	blood_glucose -- one measurement / prediction

	Output:
	Penalty
	"""
	blood_glucose = max(blood_glucose, 1)
	return 33.5200151972786 * (np.log(blood_glucose) - np.log(112.5))**2


def get_glucose_penalty_bayer(blood_glucose, target=105): 
	"""
	Get the glucose penalty for a glucose value using the Bayer (105) method

	Parameters:
	blood_glucose -- one measurement / prediction

	Output:
	Penalty
	"""
	blood_glucose = np.maximum(blood_glucose, 1)
	return 32.9170208165394 * ((np.log(blood_glucose) - np.log(target)) ** 2)


def get_glucose_penalty_cao(blood_glucose): 
	"""
	Get the glucose penalty for a glucose value using the Cao method bounded

	Parameters:
	blood_glucose -- one measurement / prediction

	Output:
	Penalty
	"""
	if blood_glucose <= 80:
		return 1.0567 * (80 - blood_glucose)**1.3378
	elif blood_glucose <= 140:
		return 0
	else:
		return 0.4607 * (blood_glucose - 140)**1.0601


def get_glucose_penalty_van_herpe(blood_glucose): 
	"""
	Get the glucose penalty for a glucose value using the van Herpe method (bounded)

	Parameters:
	blood_glucose -- one measurement / prediction

	Output:
	Penalty
	"""
	if blood_glucose < 80:
		return 7.4680 * (80 - blood_glucose)**0.6337
	elif blood_glucose <= 110:
		return 0
	else:
		return 6.1767 * (blood_glucose - 110)**0.5635





























