# # import tensorflow as tr
# # from tensorflow.keras.models import sequential
# # from tensorflow.keras.layers import LSTM,Dense,Activation


# import numpy as np
# import matplotlib.pyplot as plt

# # Sample data (square footage and prices)
# X = np.array([1000, 1500, 2000, 2500, 3000])  # Feature: square footage
# Y = np.array([150000, 200000, 250000, 300000, 350000])  # Target: price

# # Hyperparameters
# learning_rate = 0.00001
# epochs = 10000

# # Initialize weights (slope) and bias (intercept)
# m = 0
# b = 0

# # Number of training examples
# n = len(X)

# # Training the model
# for _ in range(epochs):
#     # Calculate predictions
#     predictions = m * X + b
    
#     # Calculate error
#     error = Y - predictions
    
#     # Update weights
#     m += learning_rate * (-2/n) * np.dot(X, error)
#     b += learning_rate * (-2/n) * np.sum(error)

# # Output the trained parameters
# print(f"Trained slope (m): {m}")
# print(f"Trained intercept (b): {b}")

# # Making predictions on new data
# X_new = np.array([1200, 1800, 2200])  # New square footage data
# predictions_new = m * X_new + b
# print(f"Predictions for {X_new}: {predictions_new}")

# # Optional: Plotting the results
# plt.scatter(X, Y, color='blue', label='Data Points')
# plt.plot(X, predictions, color='red', label='Regression Line')
# plt.xlabel('Square Footage')
# plt.ylabel('Price')
# plt.title('Linear Regression Example')
# plt.legend()
# plt.show()









