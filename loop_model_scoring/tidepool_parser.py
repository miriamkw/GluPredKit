#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0200, C0103, R0912, R0913, R0914, R0915
import warnings

from datetime import datetime, time, timedelta
import numpy

from pyloopkit.dose import DoseType
from pyloopkit.loop_data_manager import update
from pyloopkit.loop_math import sort_dose_lists


# %% Functions to get various data from an issue report
def get_glucose_data(glucose_data, offset=0):
    """ Load glucose values from an issue report cached_glucose dictionary

    Arguments:
    df -- the dataframe of the CGM measurements in a Tidepool export

    Output:
    2 lists in (date, glucose_value) format
    """
    dates = [parse_datetime_string(sample['time']) + timedelta(seconds=offset) for sample in glucose_data]
    glucose_values = [sample['value'] * 18.0182 if sample['units'] == 'mmol/L' else sample['value'] for sample in
                      glucose_data]
    assert len(dates) == len(glucose_values), "expected output shape to match"
    return dates, glucose_values


def parse_datetime_string(datetime_string):
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        # Add other formats here as needed
    ]

    for format in formats:
        try:
            return datetime.strptime(datetime_string, format)
        except ValueError:
            continue

    raise ValueError("Datetime string does not match any known formats")


def get_insulin_data(
        bolus_data, basal_data,
        offset=0):
    """ Load doses from an issue report cached_doses
    or normalized_insulin_doses dictionary

    Arguments:
    data -- dictionary containing cached dose information
    offset -- the offset from UTC in seconds

    Output:
    4 lists in (dose_type (DoseType enum), start_dates, end_dates,
        values (in units/insulin)) format
    """
    dose_types = [
        DoseType.from_str(
            "bolus"
        ) for _ in range(len(bolus_data))
    ]

    values = []

    start_dates = [parse_datetime_string(sample['time']) + timedelta(seconds=offset) for sample in bolus_data]
    end_dates = start_dates.copy()

    for sample in bolus_data:
        value = sample['normal']
        date = parse_datetime_string(sample['time']) + timedelta(seconds=offset)

        # For bolus doses end date is not recorded in the Tidepool API
        # start_dates.append(date)
        # end_dates.append(date)
        values.append(value)

    for sample in basal_data:
        # Basal values are stored as U/h
        value = sample['rate']

        start_date = parse_datetime_string(sample['time']) + timedelta(seconds=offset)
        end_date = start_date + timedelta(milliseconds=sample['duration'])

        start_dates.append(start_date)
        end_dates.append(end_date)

        if sample['deliveryType'] == 'temp':
            dose_type = DoseType.tempbasal
            dose_types.append(dose_type)
        else:
            dose_type = DoseType.basal
            dose_types.append(dose_type)

        values.append(value)

    assert len(dose_types) == len(start_dates) == len(end_dates) == \
           len(values), \
        "expected output shape to match"

    return dose_types, start_dates, end_dates, values


def get_carb_data(carb_data, offset=0):
    """ Load carb information from an issue report cached_carbs dictionary

    Arguments:
    data -- dictionary containing cached carb information
    offset -- the offset from UTC in seconds

    Output:
    3 lists in (carb_values, carb_start_dates, carb_absorption_times)
    format
    """
    start_dates = [parse_datetime_string(sample['time']) + timedelta(seconds=offset) for sample in carb_data]
    carb_values = [sample['nutrition']['carbohydrate']['net'] for sample in carb_data]
    absorption_times = [sample['payload']['com.loopkit.AbsorptionTime'] / 60 for sample in carb_data]

    assert len(start_dates) == len(carb_values) == len(absorption_times), \
        "expected input shapes to match"
    return start_dates, carb_values, absorption_times


def seconds_to_time(seconds):
    """ Convert seconds since midnight into a datetime.time object """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return time(int(hours), int(minutes), int(seconds))


def get_starts_and_ends_from_seconds(seconds_list):
    """ Given a list of seconds since midnight,
        convert into start and end times
    """
    starts = [seconds_to_time(seconds) for seconds in seconds_list]
    ends = [value for value in starts]
    ends.append(ends.pop(0))

    assert len(starts) == len(ends), "expected output shapes to match"

    return (starts, ends)


def get_sensitivities(data):
    """ Load insulin sensitivity schedule
        from an issue report isf_schedule dictionary

    Arguments:
    data -- dictionary containing ISF information

    Output:
    3 lists in (sensitivity_start_time, sensitivity_end_time,
                sensitivity_value (mg/dL/U)) format
    """
    seconds = [float(dict_.get("startTime")) for dict_ in data]

    (start_times, end_times) = get_starts_and_ends_from_seconds(seconds)

    values = [dict_.get("value") for dict_ in data]

    assert len(start_times) == len(end_times) == len(values), \
        "expected output shape to match"

    return start_times, end_times, values


def get_carb_ratios(data):
    """ Load carb ratio schedule
        from an issue report carb_ratio_schedule dictionary

    Arguments:
    data -- dictionary containing CR information

    Output:
    2 lists in (ratio_start_time, ratio_value (g/U)) format
    """
    seconds = [float(dict_.get("startTime")) for dict_ in data]

    start_times = get_starts_and_ends_from_seconds(seconds)[0]

    values = [dict_.get("value") for dict_ in data]

    assert len(start_times) == len(values), \
        "expected output shape to match"

    return (start_times, values)


def get_basal_schedule(data):
    """ Load basal rate schedule
        from an issue report basal_rate_schedule dictionary

    Arguments:
    data -- dictionary containing basal schedule information

    Output:
    3 lists in (rate_start_time, rate_length (minutes), rate (U/hr)) format
    """
    seconds = [float(dict_.get("startTime")) for dict_ in data]
    rate_minutes = []
    for i in range(0, len(seconds)):
        if i == len(seconds) - 1:
            rate_minutes.append(
                (seconds[i] - seconds[0]) / 60
            )
        else:
            rate_minutes.append(
                (seconds[i + 1] - seconds[i]) / 60
            )

    start_times = get_starts_and_ends_from_seconds(seconds)[0]

    values = [dict_.get("value") for dict_ in data]

    assert len(start_times) == len(rate_minutes) == len(values), \
        "expected output shapes to match"

    return start_times, values, rate_minutes


def get_target_range_schedule(data):
    """ Load target range schedule
        from an issue report "correction_range_schedule" dictionary
    """
    seconds = [float(dict_.get("startTime")) for dict_ in data]
    (start_times, end_times) = get_starts_and_ends_from_seconds(seconds)
    min_values = [float(dict_.get("value")[0]) for dict_ in data]
    max_values = [float(dict_.get("value")[1]) for dict_ in data]

    return start_times, end_times, min_values, max_values


def load_momentum_effects(data, offset=0):
    """ Load glucose momentum effects from a list """
    start_times = [
        datetime.strptime(
            dict_.get("startDate"),
            "%Y-%m-%d %H:%M:%S %z"
        ) + timedelta(seconds=offset)
        for dict_ in data
    ]
    values = [
        float(dict_.get("quantity")) for dict_ in data
    ]
    return start_times, values


def get_counteractions(data, offset=0):
    """ Load counteraction effect data from a list """
    start_times = [
        datetime.strptime(
            dict_.get("start_time"),
            "%Y-%m-%d %H:%M:%S %z"
        ) + timedelta(seconds=offset)
        for dict_ in data
    ]
    end_times = [
        datetime.strptime(
            dict_.get("end_time"),
            " %Y-%m-%d %H:%M:%S %z"
        ) + timedelta(seconds=offset)
        for dict_ in data
    ]
    values = [
        float(dict_.get("value")) for dict_ in data
    ]
    return start_times, end_times, values


def load_insulin_effects(data, offset=0):
    """ Load insulin effect data from a list """
    start_times = [
        datetime.strptime(
            dict_.get("start_time"),
            "%Y-%m-%d %H:%M:%S %z"
        ) + timedelta(seconds=offset)
        for dict_ in data
    ]
    values = [
        float(dict_.get("value")) for dict_ in data
    ]
    return start_times, values


def get_retrospective_effects(data, offset=0):
    """ Load retrospective effect data from a list """
    start_times = [
        datetime.strptime(
            dict_.get("startDate"),
            "%Y-%m-%d %H:%M:%S %z"
        ) + timedelta(seconds=offset)
        for dict_ in data
    ]
    values = [
        float(dict_.get("quantity")) for dict_ in data
    ]
    return start_times, values


def get_settings(data):
    """ Load needed settings from an issue report

    Arguments:
    data -- the parsed issue report dictionary

    Output:
    Dictionary of settings
    """
    settings = {}

    model = data.get("insulin_model")
    if not model:
        raise RuntimeError("No insulin model information found")

    if model.lower() == "humalognovologchild":
        settings["model"] = [
            data.get("insulin_action_duration") / 60,
            65
        ]
    elif model.lower() == "humalognovologadult":
        settings["model"] = [
            data.get("insulin_action_duration") / 60,
            75
        ]
    elif model.lower() == "fiasp":
        settings["model"] = [
            data.get("insulin_action_duration") / 60,
            55
        ]
    else:  # Walsh model
        settings["model"] = [
            data.get("insulin_action_duration") / 60 / 60
        ]

    momentum_interval = data.get("glucose_store").get("momentumDataInterval")
    if momentum_interval is not None:
        settings["momentum_data_interval"] = float(momentum_interval) / 60
    else:
        settings["momentum_data_interval"] = 15

    suspend_threshold = data.get("suspend_threshold")
    if suspend_threshold is not None:
        settings["suspend_threshold"] = float(suspend_threshold)
    else:
        settings["suspend_threshold"] = None

    settings["dynamic_carb_absorption_enabled"] = True
    settings["retrospective_correction_integration_interval"] = 30
    settings["recency_interval"] = 15
    settings["retrospective_correction_grouping_interval"] = 30
    settings["rate_rounder"] = 0.05
    settings["insulin_delay"] = 10
    settings["carb_delay"] = 10

    settings["default_absorption_times"] = [
        float(data.get("carb_default_absorption_times_fast")) / 60,
        float(data.get("carb_default_absorption_times_medium")) / 60,
        float(data.get("carb_default_absorption_times_slow")) / 60
    ]

    settings["max_basal_rate"] = data.get("maximum_basal_rate")
    settings["max_bolus"] = data.get("maximum_bolus")
    settings["retrospective_correction_enabled"] = data.get(
        "retrospective_correction_enabled"
    ) and data.get(
        "retrospective_correction_enabled"
    ).lower() == "true"

    return settings


# %% List management tools
def sort_by_first_list(list_1, list_2, list_3=None, list_4=None, list_5=None):
    """ Sort lists that are matched index-wise, using the first list as the
        property to sort by

    Example:
        l1: [50, 2, 3]               ->     [2, 3, 50]
        l2: [dog, cat, parrot]       ->     [cat, parrot, dog]
    """
    unsort_1 = numpy.array(list_1)
    unsort_2 = numpy.array(list_2)
    unsort_3 = numpy.array(list_3)
    unsort_4 = numpy.array(list_4)
    unsort_5 = numpy.array(list_5)

    sort_indexes = unsort_1.argsort()

    unsort_1.sort()
    list_1 = list(unsort_1)
    l2 = list(unsort_2[sort_indexes])
    if list_3:
        l3 = list(unsort_3[sort_indexes])
    else:
        l3 = []
    if list_4:
        l4 = list(unsort_4[sort_indexes])
    else:
        l4 = []
    if list_5:
        l5 = list(unsort_5[sort_indexes])
    else:
        l5 = []

    return list_1, l2, l3, l4, l5


def remove_too_new_values(
        sort_time,
        list_1, list_2, list_3=None, list_4=None, list_5=None,
        is_dose_data=False
):
    """ Remove values that occur after a certain date. This function makes the
        assumption that the date list is sorted in ascending order, and
        that all lists (if they are not None) are the same length. The first
        list must be the list with the times, unless is_dose_data is True,
        in which case the second list must contain the times.

    Arguments:
    sort_time -- the datetime after which to remove values
    """
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []

    for i in range(0, len(list_1)):
        # if this isn't dose data, use the first list to sort
        if not is_dose_data and list_1[i] <= sort_time:
            l1.append(list_1[i])
            l2.append(list_2[i])
            if list_3:
                l3.append(list_3[i])
            if list_4:
                l4.append(list_4[i])
            if list_5:
                l5.append(list_5[i])
        # otherwise, use the second list to sort
        elif is_dose_data and list_2[i] <= sort_time:
            l1.append(list_1[i])
            l2.append(list_2[i])
            if list_3:
                l3.append(list_3[i])
            if list_4:
                l4.append(list_4[i])
            if list_5:
                l5.append(list_5[i])

    return l1, l2, l3, l4, l5


def remove_too_old_values(
        sort_time,
        list_1, list_2, list_3=None, list_4=None, list_5=None,
        is_dose_data=False
):
    """ Remove values that occur before a certain date. This function makes the
        assumption that the date list is sorted in ascending order, and
        that all lists (if they are not None) are the same length. The first
        list must be the list with the times, unless is_dose_data is True,
        in which case the second list must contain the times.

    Arguments:
    sort_time -- the datetime after which to remove values
    """
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []

    for i in range(0, len(list_1)):
        # if this isn't dose data, use the first list to sort
        if not is_dose_data and list_1[i] > sort_time:
            l1.append(list_1[i])
            l2.append(list_2[i])
            if list_3:
                l3.append(list_3[i])
            if list_4:
                l4.append(list_4[i])
            if list_5:
                l5.append(list_5[i])
        # otherwise, use the second list to sort
        elif is_dose_data and list_2[i] > sort_time:
            l1.append(list_1[i])
            l2.append(list_2[i])
            if list_3:
                l3.append(list_3[i])
            if list_4:
                l4.append(list_4[i])
            if list_5:
                l5.append(list_5[i])

    return (l1, l2, l3, l4, l5)


def get_values_by_date(
        sort_time,
        list_1, list_2, list_3=None, list_4=None, list_5=None,
        is_dose_data=False
):
    """ Remove values that occur after a certain date. This function makes the
        assumption that the date list is sorted in ascending order, and
        that all lists (if they are not None) are the same length. The first
        list must be the list with the times, unless is_dose_data is True,
        in which case the second list must contain the times.

    Arguments:
    sort_time -- the datetime after which to remove values
    """
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []

    for i in range(0, len(list_1)):
        # if this isn't dose data, use the first list to sort
        if not is_dose_data and list_1[i] <= sort_time:
            l1.append(list_1[i])
            l2.append(list_2[i])
            if list_3:
                l3.append(list_3[i])
            if list_4:
                l4.append(list_4[i])
            if list_5:
                l5.append(list_5[i])
        # otherwise, use the second list to sort
        elif is_dose_data and list_2[i] <= sort_time:
            l1.append(list_1[i])
            l2.append(list_2[i])
            if list_3:
                l3.append(list_3[i])
            if list_4:
                l4.append(list_4[i])
            if list_5:
                l5.append(list_5[i])

    return (l1, l2, l3, l4, l5)


def parse_json(user_data):
    """ Sort the user data from tidepool API into lists of the different data types

    Arguments:
    user_data -- the dictionary from an API call to the Tidepool data

    Output:
    Lists of the different data types in the json (glucose_data, bolus_data, basal_data, carb_data)
    """
    glucose_data = []
    bolus_data = []
    basal_data = []
    carb_data = []

    # Sort data types into lists
    for data in user_data:
        if data['type'] == 'cbg':
            glucose_data.append(data)
        elif data['type'] == 'bolus':
            bolus_data.append(data)
        elif data['type'] == 'basal':
            basal_data.append(data)
        elif data['type'] == 'food':
            carb_data.append(data)

    return (glucose_data, bolus_data, basal_data, carb_data)


def get_offset():
    now = datetime.now()
    utcNow = datetime.utcnow()
    return int((now - utcNow).total_seconds())


# To do
# 1) a parse report method that prepares the input_dict with settings as well as parsing values and sorting them
# 2) a run method that sorts out too new values and returns prediction

def parse_report(glucose_data, bolus_data, basal_data, carb_data):
    offset = get_offset()
    input_dict = {}

    if not len(glucose_data) == 0:
        (glucose_dates, glucose_values) = get_glucose_data(
            glucose_data,
            offset
        )
        (glucose_dates, glucose_values) = sort_by_first_list(
                glucose_dates, glucose_values
            )[0:2]
        input_dict["glucose_units"] = "mg/dL"

    else:
        raise RuntimeError("No glucose information found")

    if not (len(bolus_data) == 0 or len(basal_data) == 0):
        (dose_types,
         dose_starts,
         dose_ends,
         dose_values
         ) = get_insulin_data(
            bolus_data,
            basal_data,
            offset
        )
    else:
        warnings.warn("Warning: no insulin dose information found")
        (dose_types,
         dose_starts,
         dose_ends,
         dose_values
         ) = ([], [], [], [])

    (dose_types,
     dose_starts,
     dose_ends,
     dose_values
     ) = sort_dose_lists(
            dose_types,
            dose_starts,
            dose_ends,
            dose_values
        )[0:4]
    input_dict["dose_value_units"] = "U or U/hr"

    if not len(carb_data) == 0:
        (carb_dates,
         carb_values,
         carb_absorptions
         ) = sort_by_first_list(
            *get_carb_data(
                carb_data,
                offset,
            )
        )[0:3]
    else:
        (carb_dates,
         carb_values,
         carb_absorptions
         ) = ([], [], [])

    input_dict["carb_value_units"] = "g"

    return input_dict, glucose_dates, glucose_values, dose_types, dose_starts, dose_ends, dose_values, carb_dates, carb_values, carb_absorptions


def parse_settings(input_dict, settings_dict):
    settings = get_settings(settings_dict)
    input_dict["settings_dictionary"] = settings

    if settings_dict.get(
            "insulin_sensitivity_factor_schedule"):
        (sensitivity_start_times,
         sensitivity_end_times,
         sensitivity_values
         ) = get_sensitivities(
            settings_dict.get(
                "insulin_sensitivity_factor_schedule"
            )
        )
    else:
        raise RuntimeError("No insulin sensitivity information found")

    (sensitivity_start_times,
     sensitivity_end_times,
     sensitivity_values
     ) = sort_by_first_list(
        sensitivity_start_times,
        sensitivity_end_times,
        sensitivity_values
    )[0:3]

    input_dict["sensitivity_ratio_start_times"] = sensitivity_start_times
    input_dict["sensitivity_ratio_end_times"] = sensitivity_end_times
    input_dict["sensitivity_ratio_values"] = sensitivity_values
    input_dict["sensitivity_ratio_value_units"] = "mg/dL/U"

    if settings_dict.get("carb_ratio_schedule"):
        (carb_ratio_starts,
         carb_ratio_values
         ) = get_carb_ratios(
            settings_dict.get("carb_ratio_schedule")
        )
    else:
        raise RuntimeError("No carb ratio information found")
    (carb_ratio_starts,
     carb_ratio_values
     ) = sort_by_first_list(
        carb_ratio_starts,
        carb_ratio_values
    )[0:2]

    input_dict["carb_ratio_start_times"] = carb_ratio_starts
    input_dict["carb_ratio_values"] = carb_ratio_values
    input_dict["carb_ratio_value_units"] = "g/U"

    if settings_dict.get("basal_rate_schedule"):
        (basal_rate_starts,
         basal_rate_values,
         basal_rate_minutes
         ) = get_basal_schedule(
            settings_dict.get("basal_rate_schedule")
        )
    else:
        raise RuntimeError("No basal rate information found")
    (basal_rate_starts,
     basal_rate_minutes,
     basal_rate_values
     ) = sort_by_first_list(
        basal_rate_starts,
        basal_rate_minutes,
        basal_rate_values
    )[0:3]

    input_dict["basal_rate_start_times"] = basal_rate_starts
    input_dict["basal_rate_minutes"] = basal_rate_minutes
    input_dict["basal_rate_values"] = basal_rate_values
    input_dict["basal_rate_units"] = "U/hr"

    if settings_dict.get("correction_range_schedule"):
        (target_range_starts,
         target_range_ends,
         target_range_minimum_values,
         target_range_maximum_values
         ) = get_target_range_schedule(
            settings_dict.get("correction_range_schedule")
        )
        (target_range_starts,
         target_range_ends,
         target_range_minimum_values,
         target_range_maximum_values
         ) = sort_by_first_list(
            target_range_starts,
            target_range_ends,
            target_range_minimum_values,
            target_range_maximum_values
        )[0:4]
    else:
        raise RuntimeError("No target range rate information found")

    input_dict["target_range_start_times"] = target_range_starts
    input_dict["target_range_end_times"] = target_range_ends
    input_dict["target_range_minimum_values"] = target_range_minimum_values
    input_dict["target_range_maximum_values"] = target_range_maximum_values
    input_dict["target_range_value_units"] = "mg/dL"
    input_dict["last_temporary_basal"] = []

    return input_dict

def run_prediction(input_dict, glucose_dates, glucose_values, dose_types, dose_starts, dose_ends, dose_values, carb_dates, carb_values, carb_absorptions, time_to_run):

    input_dict["time_to_calculate_at"] = time_to_run

    (glucose_dates, glucose_values) = remove_too_new_values(
        time_to_run,
        glucose_dates,
        glucose_values
    )[0:2]
    input_dict["glucose_dates"] = glucose_dates
    input_dict["glucose_values"] = glucose_values

    (dose_types,
     dose_starts,
     dose_ends,
     dose_values
     ) = remove_too_new_values(
        time_to_run,
        dose_types,
        dose_starts,
        dose_ends,
        dose_values,
        is_dose_data=True
    )[0:4]
    input_dict["dose_types"] = dose_types
    input_dict["dose_start_times"] = dose_starts
    input_dict["dose_end_times"] = dose_ends
    input_dict["dose_values"] = dose_values
    input_dict["dose_delivered_units"] = [None for i in range(len(dose_types))]

    (carb_dates, carb_values, carb_absorptions) = remove_too_new_values(
        time_to_run,
        carb_dates,
        carb_values,
        carb_absorptions
    )[0:3]

    input_dict["carb_dates"] = carb_dates
    input_dict["carb_values"] = carb_values
    input_dict["carb_absorption_times"] = carb_absorptions

    recommendations = update(
        input_dict
    )

    return recommendations



# Take an issue report and run it through the Loop algorithm
def parse_report_and_run(glucose_data, bolus_data, basal_data, carb_data, settings_dict, time_to_run=None):
    """ Get relevent information from a Loop issue report and use it to
        run PyLoopKit. Note that the predictions are based solemnly on past data!

    Arguments:
    user_data -- the dictionary from an API call to the Tidepool data
    time_to_run -- the datetime at which the prediction will be made, by default the last glucose sample in the data

    Output:
    A dictionary of all 4 effects, the predicted glucose values, and the
    recommended basal and bolus
    """

    # TODO: Refactor so that this process is not repeated for each prediction
    settings = get_settings(settings_dict)
    offset = get_offset()

    input_dict = {}

    if not len(glucose_data) == 0:
        (glucose_dates, glucose_values) = get_glucose_data(
            glucose_data,
            offset
        )
        # Time to run is the date at which the prediction will be calculated
        # We use the first glucose measurement in the list (sorted descending) if there is no input
        if time_to_run is None:
            time_to_run = glucose_dates[0]

        (glucose_dates, glucose_values) = remove_too_new_values(
            time_to_run,
            *sort_by_first_list(
                glucose_dates, glucose_values
            )[0:2]
        )[0:2]
        input_dict["glucose_dates"] = glucose_dates
        input_dict["glucose_values"] = glucose_values
        input_dict["glucose_units"] = "mg/dL"

    else:
        raise RuntimeError("No glucose information found")

    input_dict["time_to_calculate_at"] = time_to_run

    if not (len(bolus_data) == 0 or len(basal_data) == 0):
        (dose_types,
         dose_starts,
         dose_ends,
         dose_values
         ) = get_insulin_data(
            bolus_data,
            basal_data,
            offset
        )
    else:
        warnings.warn("Warning: no insulin dose information found")
        (dose_types,
         dose_starts,
         dose_ends,
         dose_values
         ) = ([], [], [], [])

    (dose_types,
     dose_starts,
     dose_ends,
     dose_values
     ) = remove_too_new_values(
        time_to_run,
        *sort_dose_lists(
            dose_types,
            dose_starts,
            dose_ends,
            dose_values
        )[0:4],
        is_dose_data=True
    )[0:4]
    input_dict["dose_types"] = dose_types
    input_dict["dose_start_times"] = dose_starts
    input_dict["dose_end_times"] = dose_ends
    input_dict["dose_values"] = dose_values
    input_dict["dose_value_units"] = "U or U/hr"
    input_dict["dose_delivered_units"] = [None for i in range(len(dose_types))]

    if not len(carb_data) == 0:
        (carb_dates,
         carb_values,
         carb_absorptions
         ) = sort_by_first_list(
            *get_carb_data(
                carb_data,
                offset,
            )
        )[0:3]
        (carb_dates, carb_values, carb_absorptions) = remove_too_new_values(
            time_to_run,
            carb_dates,
            carb_values,
            carb_absorptions
        )[0:3]
    else:
        (carb_dates,
         carb_values,
         carb_absorptions
         ) = ([], [], [])

    input_dict["carb_dates"] = carb_dates
    input_dict["carb_values"] = carb_values
    input_dict["carb_absorption_times"] = carb_absorptions
    input_dict["carb_value_units"] = "g"

    input_dict["settings_dictionary"] = settings

    if settings_dict.get(
            "insulin_sensitivity_factor_schedule"):
        (sensitivity_start_times,
         sensitivity_end_times,
         sensitivity_values
         ) = get_sensitivities(
            settings_dict.get(
                "insulin_sensitivity_factor_schedule"
            )
        )
    else:
        raise RuntimeError("No insulin sensitivity information found")

    (sensitivity_start_times,
     sensitivity_end_times,
     sensitivity_values
     ) = sort_by_first_list(
        sensitivity_start_times,
        sensitivity_end_times,
        sensitivity_values
    )[0:3]

    input_dict["sensitivity_ratio_start_times"] = sensitivity_start_times
    input_dict["sensitivity_ratio_end_times"] = sensitivity_end_times
    input_dict["sensitivity_ratio_values"] = sensitivity_values
    input_dict["sensitivity_ratio_value_units"] = "mg/dL/U"

    if settings_dict.get("carb_ratio_schedule"):
        (carb_ratio_starts,
         carb_ratio_values
         ) = get_carb_ratios(
            settings_dict.get("carb_ratio_schedule")
        )
    else:
        raise RuntimeError("No carb ratio information found")
    (carb_ratio_starts,
     carb_ratio_values
     ) = sort_by_first_list(
        carb_ratio_starts,
        carb_ratio_values
    )[0:2]

    input_dict["carb_ratio_start_times"] = carb_ratio_starts
    input_dict["carb_ratio_values"] = carb_ratio_values
    input_dict["carb_ratio_value_units"] = "g/U"

    if settings_dict.get("basal_rate_schedule"):
        (basal_rate_starts,
         basal_rate_values,
         basal_rate_minutes
         ) = get_basal_schedule(
            settings_dict.get("basal_rate_schedule")
        )
    else:
        raise RuntimeError("No basal rate information found")
    (basal_rate_starts,
     basal_rate_minutes,
     basal_rate_values
     ) = sort_by_first_list(
        basal_rate_starts,
        basal_rate_minutes,
        basal_rate_values
    )[0:3]

    input_dict["basal_rate_start_times"] = basal_rate_starts
    input_dict["basal_rate_minutes"] = basal_rate_minutes
    input_dict["basal_rate_values"] = basal_rate_values
    input_dict["basal_rate_units"] = "U/hr"

    if settings_dict.get("correction_range_schedule"):
        (target_range_starts,
         target_range_ends,
         target_range_minimum_values,
         target_range_maximum_values
         ) = get_target_range_schedule(
            settings_dict.get("correction_range_schedule")
        )
        (target_range_starts,
         target_range_ends,
         target_range_minimum_values,
         target_range_maximum_values
         ) = sort_by_first_list(
            target_range_starts,
            target_range_ends,
            target_range_minimum_values,
            target_range_maximum_values
        )[0:4]
    else:
        raise RuntimeError("No target range rate information found")

    input_dict["target_range_start_times"] = target_range_starts
    input_dict["target_range_end_times"] = target_range_ends
    input_dict["target_range_minimum_values"] = target_range_minimum_values
    input_dict["target_range_maximum_values"] = target_range_maximum_values
    input_dict["target_range_value_units"] = "mg/dL"
    input_dict["last_temporary_basal"] = []

    recommendations = update(
        input_dict
    )

    return recommendations
