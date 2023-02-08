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


def get_ideal_treatment(blood_glucose_value, end_value):
	""" Get the blood glucose trajectory given the ideal treatment

	Arguments:
	blood_glucose_value -- the blood glucose measurement at the time of reference for a forecast
	interval -- the interval between each blood glucose measuremen/prediction in minutes
	end_value -- the end blood glucose in mg/dL given a treatment action that will lead to this value

	Output:
	A list of glucose_values representing the trajectory of an ideal treatment
	"""
	glucose_values = [blood_glucose_value]

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

		if blood_glucose_value <= end_value:
			glucose_values.append(get_glucose_from_carbs(blood_glucose_value, end_value, t))
		else:
			glucose_values.append(blood_glucose_value - (1 - percent_effect_remaining(t - delay, action_duration, peak_activity_time)) * (blood_glucose_value - end_value))

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


def get_average_glucose_penalty(blood_glucose_values, type='bayer'): 
	"""
	Get the average glucose penalty for a list of values

	To do: Implement support for different penalty functions

	Parameters:
	blood_glucose_values -- a trajectory of predicted blood glucose values

	Output:
	Loss
	"""
	return np.mean(get_glucose_penalties(blood_glucose_values))


def get_glucose_penalties(blood_glucose_values, type='bayer'):
	return [get_glucose_penalty_bayer(x) for x in blood_glucose_values]


def get_glucose_penalty_bayer(blood_glucose, target=105): 
	"""
	Get the glucose penalty for a glucose value using the Bayer (105) method

	Parameters:
	blood_glucose -- one measurement / prediction

	Output:
	Penalty
	"""
	return 32.9170208165394 * (np.log(blood_glucose/target))**2

def get_glucose_penalty_van_herpe(blood_glucose, target=105): 
	"""
	Get the glucose penalty for a glucose value using the Bayer (105) method

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


