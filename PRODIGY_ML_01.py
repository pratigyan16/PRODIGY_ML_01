import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("Housing.csv") 

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict price range based on user input
def predict_price_range(area, bedrooms, bathrooms):
    # Create a DataFrame with user input
    user_input = pd.DataFrame([[area, bedrooms, bathrooms]], columns=["area", "bedrooms", "bathrooms"])

    # Assumed values for other features for worst case
    user_input["stories"] = X["stories"].min()
    user_input["mainroad"] = 0
    user_input["guestroom"] = 0
    user_input["basement"] = 0
    user_input["hotwaterheating"] = 0
    user_input["airconditioning"] = 0
    user_input["parking"] = 0
    user_input["prefarea"] = 0
    user_input["furnishingstatus_semi-furnished"] = 0
    user_input["furnishingstatus_unfurnished"] = 1

    # Predict price for worst case
    worst_case_price = model.predict(user_input)[0]

    # Assumed values for other features for best case
    user_input["stories"] = X["stories"].max()
    user_input["mainroad"] = 1
    user_input["guestroom"] = 1
    user_input["basement"] = 1
    user_input["hotwaterheating"] = 1
    user_input["airconditioning"] = 1
    user_input["parking"] = 1
    user_input["prefarea"] = 1
    user_input["furnishingstatus_semi-furnished"] = 1
    user_input["furnishingstatus_unfurnished"] = 0

    # Predict price for best case
    best_case_price = model.predict(user_input)[0]

    return worst_case_price, best_case_price

# Get user input
area = float(input("Enter area: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Predict price range
worst_price, best_price = predict_price_range(area, bedrooms, bathrooms)
print(f"Predicted price range: ${worst_price:.2f} - ${best_price:.2f}")
