import matplotlib.pyplot as plt
import numpy as np

# Define the x-axis range using linspace
x = np.linspace(0, 10, 20)

# Define the two lines (example equations)
start_val = 5

true = 0 * x + start_val
predicted = 0.6 * x + start_val

# Create the plot
plt.plot(x, true, 'k.', label='True blood glucose measurements', markersize=12)
plt.plot(x, predicted, 'b--', label='Predicted blood glucose measurements')

# Remove x-axis numbers (tick labels)
plt.xticks([])

# Adding background for target ares
plt.axhspan(70 / 18, 180 / 18, facecolor='blue', alpha=0.2)

# Define the y-axis tick locations and labels
y_ticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
plt.yticks(y_ticks)

# Add x and y axis labels
plt.xlabel('Time')
plt.ylabel('Blood glucose [mmol/L]')

# Add legend
plt.legend()

# Show the plot
plt.show()









