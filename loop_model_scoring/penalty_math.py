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

def get_penalty(reference_blood_glucose, true_blood_glucose, derived_blood_glucose):
	""" Get the penalty for a pair of true and derived blood glucose values

	Arguments:
	reference_blood_glucose -- reference blood glucose value for a forecast
	true_blood_glucose -- true blood glucose value t minutes into a forecast
	derived_blood_glucose -- predicted blood glucose value t minutes into a forecast

	Output:
	The average penalty over a pair of true and derived blood glucose
	"""
	target = 105

	ideal_glucose_values = get_ideal_treatment(reference_blood_glucose, target)
	derived_end_value = true_blood_glucose + target - derived_blood_glucose
	derived_glucose_values = get_ideal_treatment(reference_blood_glucose, derived_end_value)

	penalty_true = get_average_glucose_penalty(ideal_glucose_values)
	penalty_derived = get_average_glucose_penalty(derived_glucose_values)

	return penalty_derived - penalty_true


def get_ideal_treatment(start_value, end_value):
	""" Get the blood glucose trajectory for six hours for an ideal treatment given a start- and end-glucose value

	Arguments:
	start_value -- the blood glucose measurement at the beginning of a trajectory
	end_value -- the end blood glucose in mg/dL given a treatment action that will lead to this value

	Output:
	A list of glucose_values representing the trajectory of an ideal treatment
	"""
	glucose_values = [start_value]

	# Hard coding the insulin model parameters for now
	
	action_duration = 360.0
	peak_activity_time = 75.0
	delay = 10.0
	interval = 1

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
			glucose_values.append(start_value - (1 - percent_effect_remaining(t - delay, action_duration, peak_activity_time)) * (start_value - end_value))

	assert len(glucose_values) == n+1,\
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
	new_value = blood_glucose + (time - 10)*2

	if time <= 10:
		return blood_glucose
	elif new_value >= target:
		return target
	else:
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





























