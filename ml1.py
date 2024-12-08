import pandas as pd

# Price constants for each feature based on country and their currencies
data = {
    "Country": [
        "Pakistan", "USA", "India", "UK", "Canada", 
        "Australia", "Germany", "France", "Japan", 
        "Brazil", "South Africa"
    ],
    "Currency": [
        "PKR", "USD", "INR", "GBP", "CAD", 
        "AUD", "EUR", "EUR", "JPY", 
        "BRL", "ZAR"
    ],
    "GuestRoom": [
        30000, 35000, 25000, 40000, 37000, 
        36000, 45000, 42000, 50000, 
        28000, 32000
    ],
    "Bathroom": [
        15000, 20000, 12000, 25000, 22000, 
        23000, 27000, 26000, 28000, 
        14000, 16000
    ],
    "Room": [
        10000, 15000, 8000, 20000, 18000, 
        17000, 22000, 21000, 23000, 
        9000, 12000
    ],
    "StoreRoom": [
        3000, 5000, 2000, 10000, 7000, 
        6000, 12000, 11000, 13000, 
        2500, 4000
    ],
    "SwimmingPool": [
        20000, 30000, 15000, 50000, 35000, 
        40000, 60000, 55000, 70000, 
        12000, 18000
    ],
    "Garage": [
        10000, 15000, 8000, 20000, 16000, 
        15000, 25000, 24000, 30000, 
        6000, 8000
    ]
}

# Create a DataFrame
pricing_df = pd.DataFrame(data)

# Function to get user input safely
def get_user_input(prompt):
    return eval(input(prompt))

# Main function to run the program
def main():
    print("Select your country for house price prediction:")
    for i, country in enumerate(pricing_df['Country']):
        print(f"{i + 1}. {country}")

    country_choice = get_user_input(f"Enter the number of your choice (1-{len(pricing_df)}): ")

    if 1 <= country_choice <= len(pricing_df):
        country_row = pricing_df.iloc[country_choice - 1]
    else:
        print("Invalid choice. Exiting the program.")
        return

    print(f"You selected: {country_row['Country']}. Enter the following details for house price prediction:")

    WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
    WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
    guestRoomsNo = get_user_input("Number of guest rooms: ")
    store_rooms_no = get_user_input("Number of store rooms: ")
    swimming_pools_total = get_user_input("Number of swimming pools: ")
    garageno = get_user_input("Number of garages: ")

    # Calculate total price
    total_price = (
        (WithRoomBathroomNO + WithOutRoomBathroomNO) * country_row['Room'] +
        guestRoomsNo * country_row['GuestRoom'] +
        WithRoomBathroomNO * country_row['Bathroom'] +
        store_rooms_no * country_row['StoreRoom'] +
        swimming_pools_total * country_row['SwimmingPool'] +
        garageno * country_row['Garage']
    )

    print(f"\nPredicted House Price in {country_row['Country']}: {total_price} {country_row['Currency']}")

if __name__ == "__main__":
    main()



# # # # # # Price constants for each feature
# # # # # guestRoomPrice = 35000
# # # # # bathroomPrice = 20000
# # # # # roomPrice = 15000
# # # # # storeRoomPrice = 5000
# # # # # swimmingPoolPrice = 30000
# # # # # garagePrice = 15000

# # # # # # Function to get user input safely
# # # # # def get_user_input(prompt):
# # # # #     return eval(input(prompt))

# # # # # # Main function to run the program
# # # # # def main():
# # # # #     print("Enter the following details for house price prediction:")

# # # # #     WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
# # # # #     WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
# # # # #     guestRoomsNo = get_user_input("Number of guest rooms: ")
# # # # #     store_rooms_no = get_user_input("Number of store rooms: ")
# # # # #     swimming_pools_total = get_user_input("Number of swimming pools: ")
# # # # #     garageno = get_user_input("Number of garages: ")

# # # # #     # Calculate total price
# # # # #     total_price = (
# # # # #         (WithRoomBathroomNO + WithOutRoomBathroomNO) * roomPrice +
# # # # #         guestRoomsNo * guestRoomPrice +
# # # # #         WithRoomBathroomNO * bathroomPrice +
# # # # #         store_rooms_no * storeRoomPrice +
# # # # #         swimming_pools_total * swimmingPoolPrice +
# # # # #         garageno * garagePrice
# # # # #     )

# # # # #     print(f"\nPredicted House Price: ${total_price:.2f}")

# # # # # if __name__ == "__main__":
# # # # #     main()

# # # # # # # Example dataset (X: features, y: target variable)
# # # # # # X = [
# # # # # #     [1, 1, 1, 0, 1, 1],
# # # # # #     [2, 0, 2, 1, 1, 0],
# # # # # #     [3, 1, 1, 1, 0, 1],
# # # # # #     [1, 2, 1, 0, 0, 1]
# # # # # # ]

# # # # # # y = [95000, 130000, 150000, 85000]  # Corresponding prices

# # # # # # # Function to add a column of ones for the bias term
# # # # # # def add_bias(X):
# # # # # #     return [[1] + row for row in X]

# # # # # # # Fit the model using normal equation
# # # # # # def fit(X, y):
# # # # # #     X_b = add_bias(X)  # Add bias
# # # # # #     # Calculate theta using normal equation
# # # # # #     X_b_transpose = transpose(X_b)
# # # # # #     theta_best = mat_mult(mat_inv(mat_mult(X_b_transpose, X_b)), mat_mult(X_b_transpose, y))
# # # # # #     return theta_best

# # # # # # # Function to predict house price
# # # # # # def predict(features, theta):
# # # # # #     features_b = [1] + features  # Add bias term
# # # # # #     return dot_product(features_b, theta)

# # # # # # # Matrix multiplication
# # # # # # def mat_mult(A, B):
# # # # # #     if isinstance(B[0], list):  # B is a matrix
# # # # # #         return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
# # # # # #     else:  # B is a vector
# # # # # #         return [sum(a * b for a, b in zip(A_row, B)) for A_row in A]

# # # # # # # Matrix transpose
# # # # # # def transpose(M):
# # # # # #     return list(map(list, zip(*M)))

# # # # # # # Inverse of a 2x2 matrix
# # # # # # def mat_inv(matrix):
# # # # # #     det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
# # # # # #     if det == 0:
# # # # # #         raise ValueError("Matrix is not invertible")
# # # # # #     return [[matrix[1][1] / det, -matrix[0][1] / det],
# # # # # #             [-matrix[1][0] / det, matrix[0][0] / det]]

# # # # # # # Dot product for a vector and a matrix
# # # # # # def dot_product(vec, mat):
# # # # # #     return [sum(a * b for a, b in zip(vec, row)) for row in mat]

# # # # # # # Train the model
# # # # # # theta = fit(X, y)

# # # # # # # Main function to run the program
# # # # # # def main():
# # # # # #     print("Enter the following details for house price prediction:")

# # # # # #     WithRoomBathroomNO = eval(input("Number of rooms with bathrooms: "))
# # # # # #     WithOutRoomBathroomNO = eval(input("Number of rooms without bathrooms: "))
# # # # # #     guestRoomsNo = eval(input("Number of guest rooms: "))
# # # # # #     store_rooms_no = eval(input("Number of store rooms: "))
# # # # # #     swimming_pools_total = eval(input("Number of swimming pools: "))
# # # # # #     garageno = eval(input("Number of garages: "))

# # # # # #     # Prepare input features for prediction
# # # # # #     input_features = [
# # # # # #         WithRoomBathroomNO,
# # # # # #         WithOutRoomBathroomNO,
# # # # # #         guestRoomsNo,
# # # # # #         store_rooms_no,
# # # # # #         swimming_pools_total,
# # # # # #         garageno
# # # # # #     ]

# # # # # #     # Predict the total price
# # # # # #     predicted_price = predict(input_features, theta)

# # # # # #     print(f"\nPredicted House Price: ${predicted_price:.2f}")

# # # # # # if __name__ == "__main__":
# # # # # #     main()

# # # # # # # # Example dataset (X: features, y: target variable)
# # # # # # # X = [
# # # # # # #     [1, 1, 1, 0, 1, 1],
# # # # # # #     [2, 0, 2, 1, 1, 0],
# # # # # # #     [3, 1, 1, 1, 0, 1],
# # # # # # #     [1, 2, 1, 0, 0, 1]
# # # # # # # ]

# # # # # # # y = [95000, 130000, 150000, 85000]  # Corresponding prices

# # # # # # # # Function to add a column of ones for the bias term
# # # # # # # def add_bias(X):
# # # # # # #     return [[1] + row for row in X]

# # # # # # # # Fit the model using normal equation
# # # # # # # def fit(X, y):
# # # # # # #     X_b = add_bias(X)  # Add bias
# # # # # # #     # Calculate theta using normal equation
# # # # # # #     X_b_transpose = transpose(X_b)
# # # # # # #     theta_best = mat_mult(mat_inv(mat_mult(X_b_transpose, X_b)), mat_mult(X_b_transpose, y))
# # # # # # #     return theta_best

# # # # # # # # Function to predict house price
# # # # # # # def predict(features, theta):
# # # # # # #     features_b = [1] + features  # Add bias term
# # # # # # #     return dot_product(features_b, theta)

# # # # # # # # Matrix multiplication
# # # # # # # def mat_mult(A, B):
# # # # # # #     if isinstance(B[0], list):  # B is a matrix
# # # # # # #         return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
# # # # # # #     else:  # B is a vector
# # # # # # #         return [sum(a * b for a, b in zip(A_row, B)) for A_row in A]

# # # # # # # # Matrix transpose
# # # # # # # def transpose(M):
# # # # # # #     return list(map(list, zip(*M)))

# # # # # # # # Inverse of a 2x2 matrix
# # # # # # # def mat_inv(matrix):
# # # # # # #     det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
# # # # # # #     if det == 0:
# # # # # # #         raise ValueError("Matrix is not invertible")
# # # # # # #     return [[matrix[1][1] / det, -matrix[0][1] / det],
# # # # # # #             [-matrix[1][0] / det, matrix[0][0] / det]]

# # # # # # # # Dot product for a vector and a matrix
# # # # # # # def dot_product(vec, mat):
# # # # # # #     return [sum(a * b for a, b in zip(vec, row)) for row in mat]

# # # # # # # # Train the model
# # # # # # # theta = fit(X, y)

# # # # # # # # Main function to run the program
# # # # # # # def main():
# # # # # # #     print("Enter the following details for house price prediction:")

# # # # # # #     WithRoomBathroomNO = int(input("Number of rooms with bathrooms: "))
# # # # # # #     WithOutRoomBathroomNO = int(input("Number of rooms without bathrooms: "))
# # # # # # #     guestRoomsNo = int(input("Number of guest rooms: "))
# # # # # # #     store_rooms_no = int(input("Number of store rooms: "))
# # # # # # #     swimming_pools_total = int(input("Number of swimming pools: "))
# # # # # # #     garageno = int(input("Number of garages: "))

# # # # # # #     # Prepare input features for prediction
# # # # # # #     input_features = [
# # # # # # #         WithRoomBathroomNO,
# # # # # # #         WithOutRoomBathroomNO,
# # # # # # #         guestRoomsNo,
# # # # # # #         store_rooms_no,
# # # # # # #         swimming_pools_total,
# # # # # # #         garageno
# # # # # # #     ]

# # # # # # #     # Predict the total price
# # # # # # #     predicted_price = predict(input_features, theta)

# # # # # # #     print(f"\nPredicted House Price: ${predicted_price:.2f}")

# # # # # # # if __name__ == "__main__":
# # # # # # #     main()














# # # # # # # #import numpy as np

# # # # # # # # # Example dataset (X: features, y: target variable)
# # # # # # # # X = np.array([
# # # # # # # #     [1, 1, 1, 0, 1, 1],
# # # # # # # #     [2, 0, 2, 1, 1, 0],
# # # # # # # #     [3, 1, 1, 1, 0, 1],
# # # # # # # #     [1, 2, 1, 0, 0, 1]
# # # # # # # # ])

# # # # # # # # y = np.array([95000, 130000, 150000, 85000])  # Corresponding prices

# # # # # # # # # Function to add a column of ones for the bias term
# # # # # # # # def add_bias(X):
# # # # # # # #     return np.c_[np.ones(X.shape[0]), X]

# # # # # # # # # Fit the model using normal equation
# # # # # # # # def fit(X, y):
# # # # # # # #     X_b = add_bias(X)  # Add bias
# # # # # # # #     theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# # # # # # # #     return theta_best

# # # # # # # # # Function to predict house price
# # # # # # # # def predict(X, theta):
# # # # # # # #     X_b = add_bias(X)
# # # # # # # #     return X_b.dot(theta)

# # # # # # # # # Train the model
# # # # # # # # theta = fit(X, y)

# # # # # # # # # Main function to run the program
# # # # # # # # def main():
# # # # # # # #     print("Enter the following details for house price prediction:")

# # # # # # # #     WithRoomBathroomNO = eval(input("Number of rooms with bathrooms: "))
# # # # # # # #     WithOutRoomBathroomNO = eval(input("Number of rooms without bathrooms: "))
# # # # # # # #     guestRoomsNo = eval(input("Number of guest rooms: "))
# # # # # # # #     store_rooms_no = eval(input("Number of store rooms: "))
# # # # # # # #     swimming_pools_total = eval(input("Number of swimming pools: "))
# # # # # # # #     garageno = eval(input("Number of garages: "))

# # # # # # # #     # Prepare input features for prediction
# # # # # # # #     input_features = np.array([
# # # # # # # #         WithRoomBathroomNO,
# # # # # # # #         WithOutRoomBathroomNO,
# # # # # # # #         guestRoomsNo,
# # # # # # # #         store_rooms_no,
# # # # # # # #         swimming_pools_total,
# # # # # # # #         garageno
# # # # # # # #     ])

# # # # # # # #     # Predict the total price
# # # # # # # #     predicted_price = predict(input_features, theta)

# # # # # # # #     print(f"\nPredicted House Price: ${predicted_price:.2f}")

# # # # # # # # if __name__ == "__main__":
# # # # # # # #     main()








# # # # # # # # # import numpy as np

# # # # # # # # # # Example dataset (X: features, y: target variable)
# # # # # # # # # # Columns represent: [WithRoomBathroomNO, WithOutRoomBathroomNO, guestRoomsNo, store_rooms_no, swimming_pools_total, garageno]
# # # # # # # # # X = np.array([
# # # # # # # # #     [1, 1, 1, 0, 1, 1],
# # # # # # # # #     [2, 0, 2, 1, 1, 0],
# # # # # # # # #     [3, 1, 1, 1, 0, 1],
# # # # # # # # #     [1, 2, 1, 0, 0, 1]
# # # # # # # # # ])

# # # # # # # # # y = np.array([95000, 130000, 150000, 85000])  # Corresponding prices

# # # # # # # # # # Function to add a column of ones for the bias term
# # # # # # # # # def add_bias(X):
# # # # # # # # #     return np.c_[np.ones(X.shape[0]), X]

# # # # # # # # # # Fit the model using normal equation
# # # # # # # # # def fit(X, y):
# # # # # # # # #     X_b = add_bias(X)  # Add bias
# # # # # # # # #     theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# # # # # # # # #     return theta_best

# # # # # # # # # # Function to predict house price
# # # # # # # # # def predict(X, theta):
# # # # # # # # #     X_b = add_bias(X)
# # # # # # # # #     return X_b.dot(theta)

# # # # # # # # # # Train the model
# # # # # # # # # theta = fit(X, y)

# # # # # # # # # # Main function to run the program
# # # # # # # # # def main():
# # # # # # # # #     print("Enter the following details for house price prediction:")

# # # # # # # # #     WithRoomBathroomNO = float(input("Number of rooms with bathrooms: "))
# # # # # # # # #     WithOutRoomBathroomNO = float(input("Number of rooms without bathrooms: "))
# # # # # # # # #     guestRoomsNo = float(input("Number of guest rooms: "))
# # # # # # # # #     store_rooms_no = float(input("Number of store rooms: "))
# # # # # # # # #     swimming_pools_total = float(input("Number of swimming pools: "))
# # # # # # # # #     garageno = float(input("Number of garages: "))

# # # # # # # # #     # Prepare input features for prediction
# # # # # # # # #     input_features = np.array([
# # # # # # # # #         WithRoomBathroomNO,
# # # # # # # # #         WithOutRoomBathroomNO,
# # # # # # # # #         guestRoomsNo,
# # # # # # # # #         store_rooms_no,
# # # # # # # # #         swimming_pools_total,
# # # # # # # # #         garageno
# # # # # # # # #     ])

# # # # # # # # #     # Predict the total price
# # # # # # # # #     predicted_price = predict(input_features, theta)

# # # # # # # # #     print(f"\nPredicted House Price: ${predicted_price:.2f}")

# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     main()












# # # # # # # # # # def calculate_total_house_price(
# # # # # # # # # #     WithRoomBathroomNO, 
# # # # # # # # # #     WithOutRoomBathroomNO, 
# # # # # # # # # #     guestRoomsNo, 
# # # # # # # # # #     guestRoomPrice,  # Added guestRoomPrice parameter
# # # # # # # # # #     bathroomPrice, 
# # # # # # # # # #     roomPrice, 
# # # # # # # # # #     store_rooms_no, 
# # # # # # # # # #     storeRoomPrice, 
# # # # # # # # # #     swimming_pools_total, 
# # # # # # # # # #     swimmingPoolPrice, 
# # # # # # # # # #     gardenPrice, 
# # # # # # # # # #     garageno, 
# # # # # # # # # #     garagePrice
# # # # # # # # # # ):
# # # # # # # # # #     total_price = (
# # # # # # # # # #         (WithRoomBathroomNO * roomPrice) +
# # # # # # # # # #         (WithOutRoomBathroomNO * roomPrice) +
# # # # # # # # # #         (guestRoomsNo * guestRoomPrice) +  # Now this variable is defined
# # # # # # # # # #         (WithRoomBathroomNO * bathroomPrice) +
# # # # # # # # # #         (store_rooms_no * storeRoomPrice) +
# # # # # # # # # #         (swimming_pools_total * swimmingPoolPrice) +
# # # # # # # # # #         (gardenPrice) +
# # # # # # # # # #         (garageno * garagePrice)
# # # # # # # # # #     )
# # # # # # # # # #     return total_price

# # # # # # # # # # # Function to get user input
# # # # # # # # # # def get_user_input(prompt):
# # # # # # # # # #     return float(input(prompt))

# # # # # # # # # # # Main function to run the program
# # # # # # # # # # def main():
# # # # # # # # # #     print("Enter the following details for house price calculation:")

# # # # # # # # # #     WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
# # # # # # # # # #     WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
# # # # # # # # # #     guestRoomsNo = get_user_input("Number of guest rooms: ")
# # # # # # # # # #     guestRoomPrice = 35000
# # # # # # # # # #     bathroomPrice = 20000
# # # # # # # # # #     roomPrice = 15000
# # # # # # # # # #     store_rooms_no = get_user_input("Number of store rooms: ")
# # # # # # # # # #     storeRoomPrice = 5000
# # # # # # # # # #     swimming_pools_total = get_user_input("Number of swimming pools: ")
# # # # # # # # # #     swimmingPoolPrice = 30000
# # # # # # # # # #     gardenPrice = 10000
# # # # # # # # # #     garageno = get_user_input("Number of garages: ")
# # # # # # # # # #     garagePrice = 15000  


# # # # # # # # # #     # Calculate total price
# # # # # # # # # #     total_price = calculate_total_house_price(
# # # # # # # # # #         WithRoomBathroomNO,
# # # # # # # # # #         WithOutRoomBathroomNO,
# # # # # # # # # #         guestRoomsNo,
# # # # # # # # # #         guestRoomPrice,  # Pass the guest room price
# # # # # # # # # #         bathroomPrice,
# # # # # # # # # #         roomPrice,
# # # # # # # # # #         store_rooms_no,
# # # # # # # # # #         storeRoomPrice,
# # # # # # # # # #         swimming_pools_total,
# # # # # # # # # #         swimmingPoolPrice,
# # # # # # # # # #         gardenPrice,
# # # # # # # # # #         garageno,
# # # # # # # # # #         garagePrice
# # # # # # # # # #     )

# # # # # # # # # #     print(f"\nTotal House Price: ${total_price:.2f}")

# # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # #     main()













# # # # # # # # # # # def calculate_total_house_price(
# # # # # # # # # # #     WithRoomBathroomNO, 
# # # # # # # # # # #     WithOutRoomBathroomNO, 
# # # # # # # # # # #     guestRoomsNo, 
# # # # # # # # # # #     bathroomPrice, 
# # # # # # # # # # #     roomPrice, 
# # # # # # # # # # #     store_rooms_no, 
# # # # # # # # # # #     storeRoomPrice, 
# # # # # # # # # # #     swimming_pools_total, 
# # # # # # # # # # #     swimmingPoolPrice, 
# # # # # # # # # # #     gardenPrice, 
# # # # # # # # # # #     garageno, 
# # # # # # # # # # #     garagePrice
# # # # # # # # # # # ):
# # # # # # # # # # #     total_price = (
# # # # # # # # # # #         (WithRoomBathroomNO * roomPrice) +
# # # # # # # # # # #         (WithOutRoomBathroomNO * roomPrice) +
# # # # # # # # # # #         (guestRoomsNo * guestRoomPrice) +
# # # # # # # # # # #         (WithRoomBathroomNO * bathroomPrice) +
# # # # # # # # # # #         (store_rooms_no * storeRoomPrice) +
# # # # # # # # # # #         (swimming_pools_total * swimmingPoolPrice) +
# # # # # # # # # # #         (gardenPrice) +
# # # # # # # # # # #         (garageno * garagePrice)
# # # # # # # # # # #     )
# # # # # # # # # # #     return total_price

# # # # # # # # # # # # Function to get user input
# # # # # # # # # # # def get_user_input(prompt):
# # # # # # # # # # #     return float(input(prompt))

# # # # # # # # # # # # Main function to run the program
# # # # # # # # # # # def main():
# # # # # # # # # # #     print("Enter the following details for house price calculation:")

# # # # # # # # # # #     WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
# # # # # # # # # # #     WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
# # # # # # # # # # #     guestRoomsNo = get_user_input("Number of guest rooms: ")
# # # # # # # # # # #     bathroomPrice = 20000
# # # # # # # # # # #     roomPrice =15000
# # # # # # # # # # #     store_rooms_no = get_user_input("Number of store rooms: ")
# # # # # # # # # # #     storeRoomPrice = 5000
# # # # # # # # # # #     swimming_pools_total = get_user_input("Number of swimming pools: ")
# # # # # # # # # # #     swimmingPoolPrice = 30000
# # # # # # # # # # #     gardenPrice = 10000
# # # # # # # # # # #     garageno = get_user_input("Number of garages: ")
# # # # # # # # # # #     garagePrice=15000  

# # # # # # # # # # #     # Calculate total price
# # # # # # # # # # #     total_price = calculate_total_house_price(
# # # # # # # # # # #         WithRoomBathroomNO,
# # # # # # # # # # #         WithOutRoomBathroomNO,
# # # # # # # # # # #         guestRoomsNo,
# # # # # # # # # # #         bathroomPrice,
# # # # # # # # # # #         roomPrice,
# # # # # # # # # # #         store_rooms_no,
# # # # # # # # # # #         storeRoomPrice,
# # # # # # # # # # #         swimming_pools_total,
# # # # # # # # # # #         swimmingPoolPrice,
# # # # # # # # # # #         gardenPrice,
# # # # # # # # # # #         garageno,
# # # # # # # # # # #         garagePrice
# # # # # # # # # # #     )

# # # # # # # # # # #     print(f"\nTotal House Price: ${total_price:.2f}")

# # # # # # # # # # # main()










# # # # # # # # # # # # def calculate_total_house_price(
# # # # # # # # # # # #     WithRoomBathroomNO, 
# # # # # # # # # # # #     WithOutRoomBathroomNO, 
# # # # # # # # # # # #     guestRoomsNo, 
# # # # # # # # # # # #     bathroomPrice, 
# # # # # # # # # # # #     roomPrice, 
# # # # # # # # # # # #     store_rooms_no, 
# # # # # # # # # # # #     storeRoomPrice, 
# # # # # # # # # # # #     swimming_pools_total, 
# # # # # # # # # # # #     swimmingPoolPrice, 
# # # # # # # # # # # #     gardenPrice, 
# # # # # # # # # # # #     garageno, 
# # # # # # # # # # # #     garagePrice
# # # # # # # # # # # # ):
# # # # # # # # # # # #     total_price = (
# # # # # # # # # # # #         (WithRoomBathroomNO * roomPrice) +
# # # # # # # # # # # #         (WithOutRoomBathroomNO * roomPrice) +
# # # # # # # # # # # #         (guestRoomsNo * guestRoomPrice) +
# # # # # # # # # # # #         (WithRoomBathroomNO * bathroomPrice) +
# # # # # # # # # # # #         (store_rooms_no * storeRoomPrice) +
# # # # # # # # # # # #         (swimming_pools_total * swimmingPoolPrice) +
# # # # # # # # # # # #         (gardenPrice) +
# # # # # # # # # # # #         (garageno * garagePrice)
# # # # # # # # # # # #     )
# # # # # # # # # # # #     return total_price

# # # # # # # # # # # # # Example usage
# # # # # # # # # # # #     total_price = calculate_total_house_price(
# # # # # # # # # # # #         WithRoomBathroomNO,
# # # # # # # # # # # #         WithOutRoomBathroomNO,
# # # # # # # # # # # #         guestRoomsNo,
# # # # # # # # # # # #         bathroomPrice,
# # # # # # # # # # # #         roomPrice,
# # # # # # # # # # # #         store_rooms_no,
# # # # # # # # # # # #         storeRoomPrice,
# # # # # # # # # # # #         swimming_pools_total,
# # # # # # # # # # # #         swimmingPoolPrice,
# # # # # # # # # # # #         gardenPrice,
# # # # # # # # # # # #         garageno,
# # # # # # # # # # # #         garagePrice
# # # # # # # # # # # # )

# # # # # # # # # # # # print(f"Total House Price: ${total_price}")










# # # # # Price constants for each feature based on country
# # # # pricing = {
# # # #     "Pakistan": {
# # # #         "guestRoom": 30000,
# # # #         "bathroom": 15000,
# # # #         "room": 10000,
# # # #         "storeRoom": 3000,
# # # #         "swimmingPool": 20000,
# # # #         "garage": 10000
# # # #     },
# # # #     "America": {
# # # #         "guestRoom": 35000,
# # # #         "bathroom": 20000,
# # # #         "room": 15000,
# # # #         "storeRoom": 5000,
# # # #         "swimmingPool": 30000,
# # # #         "garage": 15000
# # # #     },
# # # #     "Others": {
# # # #         "guestRoom": 25000,
# # # #         "bathroom": 12000,
# # # #         "room": 8000,
# # # #         "storeRoom": 2000,
# # # #         "swimmingPool": 15000,
# # # #         "garage": 8000
# # # #     }
# # # # }

# # # # # Function to get user input safely
# # # # def get_user_input(prompt):
# # # #     return eval(input(prompt))

# # # # # Main function to run the program
# # # # def main():
# # # #     print("Select your country for house price prediction:")
# # # #     print("1. Pakistan")
# # # #     print("2. America")
# # # #     print("3. Others")
    
# # # #     country_choice = get_user_input("Enter the number of your choice (1/2/3): ")
    
# # # #     if country_choice == 1:
# # # #         country = "Pakistan"
# # # #     elif country_choice == 2:
# # # #         country = "America"
# # # #     elif country_choice == 3:
# # # #         country = "Others"
# # # #     else:
# # # #         print("Invalid choice. Exiting the program.")
# # # #         return

# # # #     print(f"You selected: {country}. Enter the following details for house price prediction:")

# # # #     WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
# # # #     WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
# # # #     guestRoomsNo = get_user_input("Number of guest rooms: ")
# # # #     store_rooms_no = get_user_input("Number of store rooms: ")
# # # #     swimming_pools_total = get_user_input("Number of swimming pools: ")
# # # #     garageno = get_user_input("Number of garages: ")

# # # #     # Calculate total price
# # # #     total_price = (
# # # #         (WithRoomBathroomNO + WithOutRoomBathroomNO) * pricing[country]["room"] +
# # # #         guestRoomsNo * pricing[country]["guestRoom"] +
# # # #         WithRoomBathroomNO * pricing[country]["bathroom"] +
# # # #         store_rooms_no * pricing[country]["storeRoom"] +
# # # #         swimming_pools_total * pricing[country]["swimmingPool"] +
# # # #         garageno * pricing[country]["garage"]
# # # #     )

# # # #     print(f"\nPredicted House Price in {country}: ${total_price:.2f}")

# # # # if __name__ == "__main__":
# # # #     main()

















# # # # Price constants for each feature based on country
# # # pricing = {
# # #     "Pakistan": {
# # #         "guestRoom": 30000,
# # #         "bathroom": 15000,
# # #         "room": 10000,
# # #         "storeRoom": 3000,
# # #         "swimmingPool": 20000,
# # #         "garage": 10000
# # #     },
# # #     "America": {
# # #         "guestRoom": 35000,
# # #         "bathroom": 20000,
# # #         "room": 15000,
# # #         "storeRoom": 5000,
# # #         "swimmingPool": 30000,
# # #         "garage": 15000
# # #     },
# # #     "India": {
# # #         "guestRoom": 25000,
# # #         "bathroom": 12000,
# # #         "room": 8000,
# # #         "storeRoom": 2000,
# # #         "swimmingPool": 15000,
# # #         "garage": 8000
# # #     },
# # #     "UK": {
# # #         "guestRoom": 40000,
# # #         "bathroom": 25000,
# # #         "room": 20000,
# # #         "storeRoom": 10000,
# # #         "swimmingPool": 50000,
# # #         "garage": 20000
# # #     },
# # #     "Canada": {
# # #         "guestRoom": 37000,
# # #         "bathroom": 22000,
# # #         "room": 18000,
# # #         "storeRoom": 7000,
# # #         "swimmingPool": 35000,
# # #         "garage": 16000
# # #     },
# # #     "Australia": {
# # #         "guestRoom": 36000,
# # #         "bathroom": 23000,
# # #         "room": 17000,
# # #         "storeRoom": 6000,
# # #         "swimmingPool": 40000,
# # #         "garage": 15000
# # #     },
# # #     "Germany": {
# # #         "guestRoom": 45000,
# # #         "bathroom": 27000,
# # #         "room": 22000,
# # #         "storeRoom": 12000,
# # #         "swimmingPool": 60000,
# # #         "garage": 25000
# # #     },
# # #     "France": {
# # #         "guestRoom": 42000,
# # #         "bathroom": 26000,
# # #         "room": 21000,
# # #         "storeRoom": 11000,
# # #         "swimmingPool": 55000,
# # #         "garage": 24000
# # #     },
# # #     "Japan": {
# # #         "guestRoom": 50000,
# # #         "bathroom": 28000,
# # #         "room": 23000,
# # #         "storeRoom": 13000,
# # #         "swimmingPool": 70000,
# # #         "garage": 30000
# # #     },
# # #     "Brazil": {
# # #         "guestRoom": 28000,
# # #         "bathroom": 14000,
# # #         "room": 9000,
# # #         "storeRoom": 2500,
# # #         "swimmingPool": 12000,
# # #         "garage": 6000
# # #     },
# # #     "South Africa": {
# # #         "guestRoom": 32000,
# # #         "bathroom": 16000,
# # #         "room": 12000,
# # #         "storeRoom": 4000,
# # #         "swimmingPool": 18000,
# # #         "garage": 8000
# # #     },
# # #     # Add more countries as needed...
# # #     "Italy": {
# # #         "guestRoom": 40000,
# # #         "bathroom": 24000,
# # #         "room": 19000,
# # #         "storeRoom": 10000,
# # #         "swimmingPool": 50000,
# # #         "garage": 20000
# # #     },
# # #     "Mexico": {
# # #         "guestRoom": 29000,
# # #         "bathroom": 13000,
# # #         "room": 8500,
# # #         "storeRoom": 3000,
# # #         "swimmingPool": 15000,
# # #         "garage": 7000
# # #     },
# # #     "Russia": {
# # #         "guestRoom": 31000,
# # #         "bathroom": 14000,
# # #         "room": 10000,
# # #         "storeRoom": 4000,
# # #         "swimmingPool": 16000,
# # #         "garage": 8000
# # #     },
# # #     "Spain": {
# # #         "guestRoom": 37000,
# # #         "bathroom": 20000,
# # #         "room": 15000,
# # #         "storeRoom": 7000,
# # #         "swimmingPool": 30000,
# # #         "garage": 14000
# # #     },
# # #     "Netherlands": {
# # #         "guestRoom": 42000,
# # #         "bathroom": 25000,
# # #         "room": 19000,
# # #         "storeRoom": 8000,
# # #         "swimmingPool": 35000,
# # #         "garage": 18000
# # #     },
# # #     "Sweden": {
# # #         "guestRoom": 43000,
# # #         "bathroom": 26000,
# # #         "room": 20000,
# # #         "storeRoom": 9000,
# # #         "swimmingPool": 40000,
# # #         "garage": 19000
# # #     },
# # #     "New Zealand": {
# # #         "guestRoom": 38000,
# # #         "bathroom": 22000,
# # #         "room": 18000,
# # #         "storeRoom": 6000,
# # #         "swimmingPool": 30000,
# # #         "garage": 16000
# # #     },
# # #     "Turkey": {
# # #         "guestRoom": 24000,
# # #         "bathroom": 13000,
# # #         "room": 7000,
# # #         "storeRoom": 2000,
# # #         "swimmingPool": 15000,
# # #         "garage": 5000
# # #     },
# # #     "Singapore": {
# # #         "guestRoom": 45000,
# # #         "bathroom": 27000,
# # #         "room": 22000,
# # #         "storeRoom": 12000,
# # #         "swimmingPool": 60000,
# # #         "garage": 25000
# # #     },
# # #     # Adding more countries up to at least 100
# # #     # Example structure continues...
# # # }

# # # # Function to get user input safely
# # # def get_user_input(prompt):
# # #     return eval(input(prompt))

# # # # Main function to run the program
# # # def main():
# # #     print("Select your country for house price prediction:")
# # #     print("1. Pakistan")
# # #     print("2. America")
# # #     print("3. Others")

# # #     country_choice = get_user_input("Enter the number of your choice (1/2/3): ")

# # #     if country_choice == 1:
# # #         country = "Pakistan"
# # #     elif country_choice == 2:
# # #         country = "America"
# # #     elif country_choice == 3:
# # #         print("Select a specific country:")
# # #         country_list = list(pricing["Others"].keys())
# # #         for i, country in enumerate(country_list):
# # #             print(f"{i + 1}. {country}")

# # #         specific_country_choice = get_user_input(f"Enter the number of your choice (1-{len(country_list)}): ")

# # #         if 1 <= specific_country_choice <= len(country_list):
# # #             country = country_list[specific_country_choice - 1]
# # #         else:
# # #             print("Invalid choice. Exiting the program.")
# # #             return
# # #     else:
# # #         print("Invalid choice. Exiting the program.")
# # #         return

# # #     print(f"You selected: {country}. Enter the following details for house price prediction:")

# # #     WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
# # #     WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
# # #     guestRoomsNo = get_user_input("Number of guest rooms: ")
# # #     store_rooms_no = get_user_input("Number of store rooms: ")
# # #     swimming_pools_total = get_user_input("Number of swimming pools: ")
# # #     garageno = get_user_input("Number of garages: ")

# # #     # Calculate total price
# # #     total_price = (
# # #         (WithRoomBathroomNO + WithOutRoomBathroomNO) * pricing[country]["room"] +
# # #         guestRoomsNo * pricing[country]["guestRoom"] +
# # #         WithRoomBathroomNO * pricing[country]["bathroom"] +
# # #         store_rooms_no * pricing[country]["storeRoom"] +
# # #         swimming_pools_total * pricing[country]["swimmingPool"] +
# # #         garageno * pricing[country]["garage"]
# # #     )

# # #     print(f"\nPredicted House Price in {country}: ${total_price:.2f}")

# # # if __name__ == "__main__":
# # #     main()










# # # Price constants for each feature based on country
# # pricing = {
# #     "Pakistan": {
# #         "guestRoom": 30000,
# #         "bathroom": 15000,
# #         "room": 10000,
# #         "storeRoom": 3000,
# #         "swimmingPool": 20000,
# #         "garage": 10000
# #     },
# #     "USA": {
# #         "guestRoom": 35000,
# #         "bathroom": 20000,
# #         "room": 15000,
# #         "storeRoom": 5000,
# #         "swimmingPool": 30000,
# #         "garage": 15000
# #     },
# #     "India": {
# #         "guestRoom": 25000,
# #         "bathroom": 12000,
# #         "room": 8000,
# #         "storeRoom": 2000,
# #         "swimmingPool": 15000,
# #         "garage": 8000
# #     },
# #     "UK": {
# #         "guestRoom": 40000,
# #         "bathroom": 25000,
# #         "room": 20000,
# #         "storeRoom": 10000,
# #         "swimmingPool": 50000,
# #         "garage": 20000
# #     },
# #     "Canada": {
# #         "guestRoom": 37000,
# #         "bathroom": 22000,
# #         "room": 18000,
# #         "storeRoom": 7000,
# #         "swimmingPool": 35000,
# #         "garage": 16000
# #     },
# #     "Australia": {
# #         "guestRoom": 36000,
# #         "bathroom": 23000,
# #         "room": 17000,
# #         "storeRoom": 6000,
# #         "swimmingPool": 40000,
# #         "garage": 15000
# #     },
# #     "Germany": {
# #         "guestRoom": 45000,
# #         "bathroom": 27000,
# #         "room": 22000,
# #         "storeRoom": 12000,
# #         "swimmingPool": 60000,
# #         "garage": 25000
# #     },
# #     "France": {
# #         "guestRoom": 42000,
# #         "bathroom": 26000,
# #         "room": 21000,
# #         "storeRoom": 11000,
# #         "swimmingPool": 55000,
# #         "garage": 24000
# #     },
# #     "Japan": {
# #         "guestRoom": 50000,
# #         "bathroom": 28000,
# #         "room": 23000,
# #         "storeRoom": 13000,
# #         "swimmingPool": 70000,
# #         "garage": 30000
# #     },
# #     "Brazil": {
# #         "guestRoom": 28000,
# #         "bathroom": 14000,
# #         "room": 9000,
# #         "storeRoom": 2500,
# #         "swimmingPool": 12000,
# #         "garage": 6000
# #     },
# #     "South Africa": {
# #         "guestRoom": 32000,
# #         "bathroom": 16000,
# #         "room": 12000,
# #         "storeRoom": 4000,
# #         "swimmingPool": 18000,
# #         "garage": 8000
# #     },
# #     "Italy": {
# #         "guestRoom": 40000,
# #         "bathroom": 24000,
# #         "room": 19000,
# #         "storeRoom": 10000,
# #         "swimmingPool": 50000,
# #         "garage": 20000
# #     },
# #     "Mexico": {
# #         "guestRoom": 29000,
# #         "bathroom": 13000,
# #         "room": 8500,
# #         "storeRoom": 3000,
# #         "swimmingPool": 15000,
# #         "garage": 7000
# #     },
# #     "Russia": {
# #         "guestRoom": 31000,
# #         "bathroom": 14000,
# #         "room": 10000,
# #         "storeRoom": 4000,
# #         "swimmingPool": 16000,
# #         "garage": 8000
# #     },
# #     "Spain": {
# #         "guestRoom": 37000,
# #         "bathroom": 20000,
# #         "room": 15000,
# #         "storeRoom": 7000,
# #         "swimmingPool": 30000,
# #         "garage": 14000
# #     },
# #     "Netherlands": {
# #         "guestRoom": 42000,
# #         "bathroom": 25000,
# #         "room": 19000,
# #         "storeRoom": 8000,
# #         "swimmingPool": 35000,
# #         "garage": 18000
# #     },
# #     "Sweden": {
# #         "guestRoom": 43000,
# #         "bathroom": 26000,
# #         "room": 20000,
# #         "storeRoom": 9000,
# #         "swimmingPool": 40000,
# #         "garage": 19000
# #     },
# #     "New Zealand": {
# #         "guestRoom": 38000,
# #         "bathroom": 22000,
# #         "room": 18000,
# #         "storeRoom": 6000,
# #         "swimmingPool": 30000,
# #         "garage": 16000
# #     },
# #     "Turkey": {
# #         "guestRoom": 24000,
# #         "bathroom": 13000,
# #         "room": 7000,
# #         "storeRoom": 2000,
# #         "swimmingPool": 15000,
# #         "garage": 5000
# #     },
# #     "Singapore": {
# #         "guestRoom": 45000,
# #         "bathroom": 27000,
# #         "room": 22000,
# #         "storeRoom": 12000,
# #         "swimmingPool": 60000,
# #         "garage": 25000
# #     },
# #     "Argentina": {
# #         "guestRoom": 27000,
# #         "bathroom": 12500,
# #         "room": 8000,
# #         "storeRoom": 2500,
# #         "swimmingPool": 13000,
# #         "garage": 5000
# #     },
# #     "Chile": {
# #         "guestRoom": 29000,
# #         "bathroom": 13000,
# #         "room": 8500,
# #         "storeRoom": 3000,
# #         "swimmingPool": 15000,
# #         "garage": 6000
# #     },
# #     "Saudi Arabia": {
# #         "guestRoom": 32000,
# #         "bathroom": 15000,
# #         "room": 12000,
# #         "storeRoom": 5000,
# #         "swimmingPool": 20000,
# #         "garage": 10000
# #     },
# #     "Thailand": {
# #         "guestRoom": 25000,
# #         "bathroom": 12000,
# #         "room": 9000,
# #         "storeRoom": 3000,
# #         "swimmingPool": 14000,
# #         "garage": 7000
# #     },
# #     "Philippines": {
# #         "guestRoom": 23000,
# #         "bathroom": 11000,
# #         "room": 7500,
# #         "storeRoom": 2000,
# #         "swimmingPool": 12000,
# #         "garage": 6000
# #     },
# #     "Vietnam": {
# #         "guestRoom": 22000,
# #         "bathroom": 10000,
# #         "room": 7000,
# #         "storeRoom": 1500,
# #         "swimmingPool": 11000,
# #         "garage": 5000
# #     },
# #     "Egypt": {
# #         "guestRoom": 24000,
# #         "bathroom": 13000,
# #         "room": 8000,
# #         "storeRoom": 2500,
# #         "swimmingPool": 15000,
# #         "garage": 6000
# #     },
# #     "Malaysia": {
# #         "guestRoom": 26000,
# #         "bathroom": 14000,
# #         "room": 9000,
# #         "storeRoom": 3000,
# #         "swimmingPool": 16000,
# #         "garage": 7000
# #     },
# #     "Colombia": {
# #         "guestRoom": 28000,
# #         "bathroom": 13500,
# #         "room": 8500,
# #         "storeRoom": 2800,
# #         "swimmingPool": 15500,
# #         "garage": 6500
# #     },
# #     "Portugal": {
# #         "guestRoom": 36000,
# #         "bathroom": 20000,
# #         "room": 18000,
# #         "storeRoom": 8000,
# #         "swimmingPool": 30000,
# #         "garage": 16000
# #     },
# #     "Ireland": {
# #         "guestRoom": 41000,
# #         "bathroom": 24000,
# #         "room": 19000,
# #         "storeRoom": 10000,
# #         "swimmingPool": 45000,
# #         "garage": 19000
# #     },
# #     "Greece": {
# #         "guestRoom": 38000,
# #         "bathroom": 23000,
# #         "room": 17000,
# #         "storeRoom": 6000,
# #         "swimmingPool": 40000,
# #         "garage": 15000
# #     },
# #     "Czech Republic": {
# #         "guestRoom": 34000,
# #         "bathroom": 20000,
# #         "room": 16000,
# #         "storeRoom": 7000,
# #         "swimmingPool": 30000,
# #         "garage": 14000
# #     },
# #     "Norway": {
# #         "guestRoom": 46000,
# #         "bathroom": 27000,
# #         "room": 23000,
# #         "storeRoom": 12000,
# #         "swimmingPool": 65000,
# #         "garage": 25000
# #     },
# #     "Finland": {
# #         "guestRoom": 44000,
# #         "bathroom": 26000,
# #         "room": 21000,
# #         "storeRoom": 11000,
# #         "swimmingPool": 60000,
# #         "garage": 24000
# #     },
# #     "Denmark": {
# #         "guestRoom": 43000,
# #         "bathroom": 25000,
# #         "room": 20000,
# #         "storeRoom": 9000,
# #         "swimmingPool": 55000,
# #         "garage": 23000
# #     },
# #     "Switzerland": {
# #         "guestRoom": 50000,
# #         "bathroom": 28000,
# #         "room": 25000,
# #         "storeRoom": 15000,
# #         "swimmingPool": 70000,
# #         "garage": 30000
# #     },
# #     "Hungary": {
# #         "guestRoom": 31000,
# #         "bathroom": 15000,
# #         "room": 11000,
# #         "storeRoom": 4000,
# #         "swimmingPool": 16000,
# #         "garage": 7000
# #     },
# #     "Romania": {
# #         "guestRoom": 29000,
# #         "bathroom": 14000,
# #         "room": 10000,
# #         "storeRoom": 3000,
# #         "swimmingPool": 15000,
# #         "garage": 6000
# #     },
# #     "Slovakia": {
# #         "guestRoom": 31000,
# #         "bathroom": 15000,
# #         "room": 11000,
# #         "storeRoom": 4000,
# #         "swimmingPool": 16000,
# #         "garage": 7000
# #     },
# #     "Bulgaria": {
# #         "guestRoom": 28000,
# #         "bathroom": 13000,
# #         "room": 9000,
# #         "storeRoom": 2500,
# #         "swimmingPool": 14000,
# #         "garage": 5000
# #     },
# #     "Ukraine": {
# #         "guestRoom": 29000,
# #         "bathroom": 14000,
# #         "room": 10000,
# #         "storeRoom": 3000,
# #         "swimmingPool": 15000,
# #         "garage": 6000
# #     },
# #     "Israel": {
# #         "guestRoom": 40000,
# #         "bathroom": 25000,
# #         "room": 20000,
# #         "storeRoom": 10000,
# #         "swimmingPool": 50000,
# #         "garage": 20000
# #     },
# #     "Iceland": {
# #         "guestRoom": 42000,
# #         "bathroom": 26000,
# #         "room": 21000,
# #         "storeRoom": 11000,
# #         "swimmingPool": 55000,
# #         "garage": 24000
# #     },
# #     "Lithuania": {
# #         "guestRoom": 29000,
# #         "bathroom": 14000,
# #         "room": 10000,
# #         "storeRoom": 3000,
# #         "swimmingPool": 15000,
# #         "garage": 6000
# #     },
# #     "Latvia": {
# #         "guestRoom": 28000,
# #         "bathroom": 13000,
# #         "room": 9000,
# #         "storeRoom": 2500,
# #         "swimmingPool": 14000,
# #         "garage": 5000
# #     },
# #     "Estonia": {
# #         "guestRoom": 27000,
# #         "bathroom": 12000,
# #         "room": 8500,
# #         "storeRoom": 2000,
# #         "swimmingPool": 13000,
# #         "garage": 4500
# #     },
# #     "Moldova": {
# #         "guestRoom": 26000,
# #         "bathroom": 11000,
# #         "room": 8000,
# #         "storeRoom": 2000,
# #         "swimmingPool": 12000,
# #         "garage": 4000
# #     },
# #     "Armenia": {
# #         "guestRoom": 25000,
# #         "bathroom": 10000,
# #         "room": 7500,
# #         "storeRoom": 1500,
# #         "swimmingPool": 11000,
# #         "garage": 3500
# #     },
# #     "Georgia": {
# #         "guestRoom": 24000,
# #         "bathroom": 9500,
# #         "room": 7000,
# #         "storeRoom": 1500,
# #         "swimmingPool": 10000,
# #         "garage": 3000
# #     },
# #     "Kyrgyzstan": {
# #         "guestRoom": 23000,
# #         "bathroom": 9000,
# #         "room": 6500,
# #         "storeRoom": 1500,
# #         "swimmingPool": 9000,
# #         "garage": 2500
# #     },
# #     "Kazakhstan": {
# #         "guestRoom": 22000,
# #         "bathroom": 8500,
# #         "room": 6000,
# #         "storeRoom": 1500,
# #         "swimmingPool": 8500,
# #         "garage": 2000
# #     },
# #     "Tajikistan": {
# #         "guestRoom": 21000,
# #         "bathroom": 8000,
# #         "room": 5500,
# #         "storeRoom": 1500,
# #         "swimmingPool": 8000,
# #         "garage": 1500
# #     },
# #     "Turkmenistan": {
# #         "guestRoom": 20000,
# #         "bathroom": 7500,
# #         "room": 5000,
# #         "storeRoom": 1500,
# #         "swimmingPool": 7500,
# #         "garage": 1000
# #     },
# #     "Uzbekistan": {
# #         "guestRoom": 19000,
# #         "bathroom": 7000,
# #         "room": 4500,
# #         "storeRoom": 1500,
# #         "swimmingPool": 7000,
# #         "garage": 800
# #     },
# #     "Afghanistan": {
# #         "guestRoom": 18000,
# #         "bathroom": 6500,
# #         "room": 4000,
# #         "storeRoom": 1500,
# #         "swimmingPool": 6500,
# #         "garage": 700
# #     },
# #     "Syria": {
# #         "guestRoom": 17000,
# #         "bathroom": 6000,
# #         "room": 3500,
# #         "storeRoom": 1500,
# #         "swimmingPool": 6000,
# #         "garage": 600
# #     },
# #     "Lebanon": {
# #         "guestRoom": 16000,
# #         "bathroom": 5500,
# #         "room": 3000,
# #         "storeRoom": 1500,
# #         "swimmingPool": 5500,
# #         "garage": 500
# #     },
# #     "Jordan": {
# #         "guestRoom": 15000,
# #         "bathroom": 5000,
# #         "room": 2500,
# #         "storeRoom": 1500,
# #         "swimmingPool": 5000,
# #         "garage": 400
# #     },
# #     "Oman": {
# #         "guestRoom": 14000,
# #         "bathroom": 4500,
# #         "room": 2000,
# #         "storeRoom": 1500,
# #         "swimmingPool": 4500,
# #         "garage": 300
# #     },
# #     "Qatar": {
# #         "guestRoom": 13000,
# #         "bathroom": 4000,
# #         "room": 1500,
# #         "storeRoom": 1500,
# #         "swimmingPool": 4000,
# #         "garage": 200
# #     },
# # }

# # # Function to get user input safely
# # def get_user_input(prompt):
# #     return eval(input(prompt))

# # # Main function to run the program
# # def main():
# #     print("Select your country for house price prediction:")
# #     country_list = list(pricing.keys())
# #     for i, country in enumerate(country_list):
# #         print(f"{i + 1}. {country}")

# #     country_choice = get_user_input(f"Enter the number of your choice (1-{len(country_list)}): ")

# #     if 1 <= country_choice <= len(country_list):
# #         country = country_list[country_choice - 1]
# #     else:
# #         print("Invalid choice. Exiting the program.")
# #         return

# #     print(f"You selected: {country}. Enter the following details for house price prediction:")

# #     WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
# #     WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
# #     guestRoomsNo = get_user_input("Number of guest rooms: ")
# #     store_rooms_no = get_user_input("Number of store rooms: ")
# #     swimming_pools_total = get_user_input("Number of swimming pools: ")
# #     garageno = get_user_input("Number of garages: ")

# #     # Calculate total price
# #     total_price = (
# #         (WithRoomBathroomNO + WithOutRoomBathroomNO) * pricing[country]["room"] +
# #         guestRoomsNo * pricing[country]["guestRoom"] +
# #         WithRoomBathroomNO * pricing[country]["bathroom"] +
# #         store_rooms_no * pricing[country]["storeRoom"] +
# #         swimming_pools_total * pricing[country]["swimmingPool"] +
# #         garageno * pricing[country]["garage"]
# #     )

# #     print(f"\nPredicted House Price in {country}: ${total_price:.2f}")

# # if __name__ == "__main__":
# #     main()









# # Price constants for each feature based on country and their currencies
# pricing = {
#     "Pakistan": {
#         "currency": "PKR",
#         "guestRoom": 30000,
#         "bathroom": 15000,
#         "room": 10000,
#         "storeRoom": 3000,
#         "swimmingPool": 20000,
#         "garage": 10000
#     },
#     "USA": {
#         "currency": "USD",
#         "guestRoom": 35000,
#         "bathroom": 20000,
#         "room": 15000,
#         "storeRoom": 5000,
#         "swimmingPool": 30000,
#         "garage": 15000
#     },
#     "India": {
#         "currency": "INR",
#         "guestRoom": 25000,
#         "bathroom": 12000,
#         "room": 8000,
#         "storeRoom": 2000,
#         "swimmingPool": 15000,
#         "garage": 8000
#     },
#     "UK": {
#         "currency": "GBP",
#         "guestRoom": 40000,
#         "bathroom": 25000,
#         "room": 20000,
#         "storeRoom": 10000,
#         "swimmingPool": 50000,
#         "garage": 20000
#     },
#     "Canada": {
#         "currency": "CAD",
#         "guestRoom": 37000,
#         "bathroom": 22000,
#         "room": 18000,
#         "storeRoom": 7000,
#         "swimmingPool": 35000,
#         "garage": 16000
#     },
#     "Australia": {
#         "currency": "AUD",
#         "guestRoom": 36000,
#         "bathroom": 23000,
#         "room": 17000,
#         "storeRoom": 6000,
#         "swimmingPool": 40000,
#         "garage": 15000
#     },
#     "Germany": {
#         "currency": "EUR",
#         "guestRoom": 45000,
#         "bathroom": 27000,
#         "room": 22000,
#         "storeRoom": 12000,
#         "swimmingPool": 60000,
#         "garage": 25000
#     },
#     "France": {
#         "currency": "EUR",
#         "guestRoom": 42000,
#         "bathroom": 26000,
#         "room": 21000,
#         "storeRoom": 11000,
#         "swimmingPool": 55000,
#         "garage": 24000
#     },
#     "Japan": {
#         "currency": "JPY",
#         "guestRoom": 50000,
#         "bathroom": 28000,
#         "room": 23000,
#         "storeRoom": 13000,
#         "swimmingPool": 70000,
#         "garage": 30000
#     },
#     "Brazil": {
#         "currency": "BRL",
#         "guestRoom": 28000,
#         "bathroom": 14000,
#         "room": 9000,
#         "storeRoom": 2500,
#         "swimmingPool": 12000,
#         "garage": 6000
#     },
#     "South Africa": {
#         "currency": "ZAR",
#         "guestRoom": 32000,
#         "bathroom": 16000,
#         "room": 12000,
#         "storeRoom": 4000,
#         "swimmingPool": 18000,
#         "garage": 8000
#     },
#     # Add more countries as needed...
# }

# # Function to get user input safely
# def get_user_input(prompt):
#     return eval(input(prompt))

# # Main function to run the program
# def main():
#     print("Select your country for house price prediction:")
#     country_list = list(pricing.keys())
#     for i, country in enumerate(country_list):
#         print(f"{i + 1}. {country}")

#     country_choice = get_user_input(f"Enter the number of your choice (1-{len(country_list)}): ")

#     if 1 <= country_choice <= len(country_list):
#         country = country_list[country_choice - 1]
#     else:
#         print("Invalid choice. Exiting the program.")
#         return

#     print(f"You selected: {country}. Enter the following details for house price prediction:")

#     WithRoomBathroomNO = get_user_input("Number of rooms with bathrooms: ")
#     WithOutRoomBathroomNO = get_user_input("Number of rooms without bathrooms: ")
#     guestRoomsNo = get_user_input("Number of guest rooms: ")
#     store_rooms_no = get_user_input("Number of store rooms: ")
#     swimming_pools_total = get_user_input("Number of swimming pools: ")
#     garageno = get_user_input("Number of garages: ")

#     # Calculate total price
#     total_price = (
#         (WithRoomBathroomNO + WithOutRoomBathroomNO) * pricing[country]["room"] +
#         guestRoomsNo * pricing[country]["guestRoom"] +
#         WithRoomBathroomNO * pricing[country]["bathroom"] +
#         store_rooms_no * pricing[country]["storeRoom"] +
#         swimming_pools_total * pricing[country]["swimmingPool"] +
#         garageno * pricing[country]["garage"]
#     )

#     print(f"\nPredicted House Price in {country}: {total_price} {pricing[country]['currency']}")

# if __name__ == "__main__":
#     main()
