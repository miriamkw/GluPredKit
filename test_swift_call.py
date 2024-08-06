import ctypes
import pandas as pd
import numpy as np
from glupredkit.parsers.tidepool import Parser

start_date = pd.to_datetime('2024-06-22')
end_date = pd.to_datetime('2024-06-25')
start_test_date = pd.to_datetime('2024-06-23T16:00:11Z')
username = "miriamkwolff@outlook.com"
password = "#UVqdyU83d3jzXxK"

max_errors = []
mean_errors = []
for i in range(200):
    start_test_date = start_test_date - pd.Timedelta(minutes=5)

    json_data_parsed = Parser().get_json_from_parsed_df(start_date, end_date, username, password, start_test_date,
                                                        basal=0.75, isf=66.6, cr=9)
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_data_parsed)

    json_bytes_parsed = json_data_parsed.encode('utf-8')  # Convert JSON string to bytes

    recommendation_settings = Parser().getRecommendationSettings()
    json_data = Parser().get_json_from_raw_data(start_date, end_date, username, password, start_test_date, basal=0.75,
                                                isf=66.6, cr=9, recommendation_settings=recommendation_settings)
    json_bytes = json_data.encode('utf-8')  # Convert JSON string to bytes

    # Load the shared library
    swift_lib = ctypes.CDLL('./libLoopAlgorithmToPython.dylib')

    # Specify the argument types and return type of the Swift function
    swift_lib.generatePrediction.argtypes = [ctypes.c_char_p]
    swift_lib.generatePrediction.restype = ctypes.POINTER(ctypes.c_double)

    # Prepare a variable to receive the length of the array
    length = 72

    # Call the Swift function
    result = swift_lib.generatePrediction(json_bytes)
    glucose_array = [result[i] for i in range(length)] # Read the array from the returned pointer

    result = swift_lib.generatePrediction(json_bytes_parsed)
    glucose_array_parsed = [result[i] for i in range(length)] # Read the array from the returned pointer

    # Specify the argument types and return type of the Swift function
    swift_lib.getPredictionDates.argtypes = [ctypes.c_char_p]
    swift_lib.getPredictionDates.restype = ctypes.c_char_p

    # Call the Swift function
    result = swift_lib.getPredictionDates(json_bytes).decode('utf-8')
    date_list = result.split(',')[:-1]

    result = swift_lib.getPredictionDates(json_bytes_parsed).decode('utf-8')
    date_list_parsed = result.split(',')[:-1]

    max_error = 0
    all_errors = []
    for j in range(length):
        #print(f'T: {date_list[i]}, {date_list_parsed[i]} GLUCOSE {glucose_array[i] - glucose_array_parsed[i]}')
        all_errors += [np.abs(glucose_array[j] - glucose_array_parsed[j])]
        if np.abs(glucose_array[j] - glucose_array_parsed[j]) > max_error:
            max_error = np.abs(glucose_array[j] - glucose_array_parsed[j])

    swift_lib.getActiveCarbs.argtypes = [ctypes.c_char_p]
    swift_lib.getActiveCarbs.restype = ctypes.c_double

    swift_lib.getActiveInsulin.argtypes = [ctypes.c_char_p]
    swift_lib.getActiveInsulin.restype = ctypes.c_double

    # Call the Swift functions
    result_active_carbs = swift_lib.getActiveCarbs(json_bytes)
    result_active_insulin = swift_lib.getActiveInsulin(json_bytes)

    # Read the active carbohydrates
    print(f"The result from getActiveCarbs is: {result_active_carbs}")

    # Read the active insulin
    print(f"The result from getActiveInsulin is: {result_active_insulin}")

    mean_error = np.mean(all_errors)
    print("MEAN ERROR", mean_error)
    print("MAX ERROR", max_error)
    print("MAX ERROR mmol/L", max_error / 18)
    print("Iteration number: ", i)
    max_errors += [max_error]
    mean_errors += [mean_error]

print('Average max errors', np.mean(max_errors))
print('Max max errors', np.max(max_errors))
print('Mean error', np.mean(mean_errors))

