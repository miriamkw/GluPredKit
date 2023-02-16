from loop_model_scoring.penalty_math import (
    get_ideal_treatment,
    get_average_glucose_penalty,
    get_glucose_penalties
)
import matplotlib.pyplot as plt
import numpy as np

# Recreating the examples in the Loop Model Scoring documentation.
reference_blood_glucose = 90
true_blood_glucose = 90
derived_blood_glucose = 140
target = 105

ideal_glucose_values = get_ideal_treatment(reference_blood_glucose, target)
derived_end_value = true_blood_glucose + target - derived_blood_glucose
derived_glucose_values = get_ideal_treatment(reference_blood_glucose, derived_end_value)


# Plot the treatment decisions made using derived vs true values
t = np.arange(0.0, 361.0, 1.0)
fig, ax = plt.subplots()
ax.plot(t, derived_glucose_values, label='Derived')
ax.plot(t, ideal_glucose_values, label='True')

ax.set(xlabel='Time (minutes)', ylabel='Simulated Blood Glucose (mg/dL)',
       title='Treatment Decisions')
ax.grid()
plt.legend(loc='best')

plt.show()


# Plot the penalty of simulated true blood glucose traces
derived_penalties = get_glucose_penalties(derived_glucose_values)
true_penalties = get_glucose_penalties(ideal_glucose_values)

fig, ax = plt.subplots()
plt.axhline(y = np.mean(derived_penalties), color='g', linestyle = '--')
plt.axhline(y = np.mean(true_penalties), color='g', linestyle = '--')
ax.plot(t, derived_penalties, label='Derived')
ax.plot(t, true_penalties, label='True')

ax.set(xlabel='Time (minutes)', ylabel='Penalty',
       title='Penalty of Simulated True Blood Glucose Traces')
ax.grid()
plt.legend(loc='best')

plt.show()


