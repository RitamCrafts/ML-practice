import pandas as pd
import joblib
from scipy.sparse import hstack

# Load trained model and encoder
model = joblib.load("car_price_model.pkl")
encoder = joblib.load("car_price_encoder.pkl")


# Take input
year = int(input("Enter car year: "))
km_driven = int(input("Enter kilometers driven: "))


# Fuel selection
print("\nFuel:")
print("1. Petrol")
print("2. Diesel")
print("3. CNG")
print("4. LPG")
print("5. Electric")

fuel_choice = int(input("Select: "))

fuel_list = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
fuel = fuel_list[fuel_choice - 1]


# Seller type selection
print("\nSeller Type:")
print("1. Individual")
print("2. Dealer")
print("3. Trustmark Dealer")

seller_choice = int(input("Select: "))

seller_list = ["Individual", "Dealer", "Trustmark Dealer"]
seller_type = seller_list[seller_choice - 1]


# Transmission selection
print("\nTransmission:")
print("1. Manual")
print("2. Automatic")

transmission_choice = int(input("Select: "))

transmission_list = ["Manual", "Automatic"]
transmission = transmission_list[transmission_choice - 1]


# Owner selection
print("\nOwner:")
print("1. First Owner")
print("2. Second Owner")
print("3. Third Owner")
print("4. Fourth & Above Owner")
print("5. Test Drive Car")

owner_choice = int(input("Select: "))

owner_list = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car"
]

owner = owner_list[owner_choice - 1]


# Car name
name = input("\nEnter full car name: ")

# Extract brand and model exactly like training
brand = name.split()[0]
model_name = name.split()[1]


# Create one row of data
data = pd.DataFrame([{
    "year": year,
    "km_driven": km_driven,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "brand": brand,
    "model": model_name
}])


# Categorical columns
categorical_columns = [
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "brand",
    "model"
]


# Encode categorical data
encoded = encoder.transform(data[categorical_columns])


# Get numeric data
numeric = data[["year", "km_driven"]].to_numpy()


# Combine numeric + encoded data
final_data = hstack([numeric, encoded])


# Predict
prediction = model.predict(final_data)


print("\n-----------------------------------")
print(f"Predicted price: Rs.{prediction[0]:.0f}")
print("-----------------------------------")