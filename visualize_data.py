import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv('vector_field_data_with_pid.csv')

# Create a figure with four subplots (removing the curve fitting subplot)
fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot X Position vs Time
ax1.plot(data['Time'], data['X'])
ax1.set_xlabel('Time')
ax1.set_ylabel('X Position')
ax1.set_title('X Position vs Time')
ax1.grid(True)

# Plot Input Acceleration vs Time
ax2.plot(data['Time'], data['Alpha'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Input Acceleration')
ax2.set_title('Input Acceleration vs Time')
ax2.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
