import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Example data: Time (e.g., hours) and corresponding demand
time = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # Hours of the day
demand = np.array([50, 70, 80, 110, 130, 150, 180, 200])  # Passengers

# Plot the demand data
plt.scatter(time, demand, color='blue', label='Observed Demand')
plt.xlabel('Time of Day (Hours)')
plt.ylabel('Demand (Passengers)')
plt.title('Demand over Time')
plt.legend()
plt.show()

# Add a column of ones for the intercept term
A = np.vstack([time, np.ones(len(time))]).T

# Calculate the least squares solution
slope, intercept = inv(A.T @ A) @ A.T @ demand

# Predicted demand
predicted_demand = slope * time + intercept

# Plot the fitted line
plt.plot(time, predicted_demand, color='red', label='Fitted Model')
plt.scatter(time, demand, color='blue', label='Observed Demand')
plt.xlabel('Time of Day (Hours)')
plt.ylabel('Demand (Passengers)')
plt.title('Demand Model using Least Squares')
plt.legend()
plt.show()

# Display the slope and intercept
print(f"Slope: {slope}, Intercept: {intercept}")

# Define total buses available and capacity
total_buses = 10  # Total buses available
bus_capacity = 20  # Bus capacity

# Calculate required buses for each time period based on the predicted demand
required_buses = np.ceil(predicted_demand / bus_capacity).astype(int)

# Ensure the total number of buses does not exceed the available buses
allocation = np.minimum(required_buses, total_buses)

# Print allocation
print("Predicted Demand:", predicted_demand)
print("Required Buses:", required_buses)
print("Final Bus Allocation:", allocation)
