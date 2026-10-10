import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# LOAD MODEL AND ENCODER
# --------------------------------------------------

model = joblib.load("phone_price_model.pkl")
encoder = joblib.load("Mobiles-Dataset-Encoder.pkl")


# --------------------------------------------------
# BRAND SELECTION
# --------------------------------------------------

brand_options = encoder.categories_[0]

print("\nSelect Brand:")

for i in range(0, len(brand_options), 2):
    left = f"{i + 1}. {brand_options[i]}"

    if i + 1 < len(brand_options):
        right = f"{i + 2}. {brand_options[i + 1]}"
    else:
        right = ""

    print(f"{left:<25} {right}")

brand_choice = int(input("Enter choice: "))

if not 1 <= brand_choice <= len(brand_options):
    raise ValueError("Invalid brand choice.")

brand = brand_options[brand_choice - 1]


# --------------------------------------------------
# PHONE DETAILS
# --------------------------------------------------

model_name = input("\nModel Name: ").strip()

rom = float(input("ROM (GB): "))
ram = float(input("RAM (GB): "))

weight_input = input(
    "Mobile Weight (g) [Enter if unknown]: "
).strip()

mobile_weight = float(weight_input) if weight_input else 0.0


# --------------------------------------------------
# FRONT SELFIE CAMERA
# Example: 8MP or 8
# --------------------------------------------------

front_camera_input = input(
    "Front Selfie Camera (example: 8MP or 8): "
).strip()

front_number = pd.to_numeric(
    pd.Series([front_camera_input])
    .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
    errors="coerce"
).iloc[0]

if pd.isna(front_number):
    raise ValueError("Enter a valid front selfie camera value.")

front_camera = float(front_number)


# --------------------------------------------------
# MAIN REAR CAMERA
# Example: 50MP or 50MP + 12MP
# Only the main camera value is used.
# --------------------------------------------------

back_camera = input(
    "Main Rear Camera (example: 50MP or 50MP + 12MP): "
).strip()

main_camera_input = back_camera.split("+")[0].strip()

main_camera_number = pd.to_numeric(
    pd.Series([main_camera_input])
    .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
    errors="coerce"
).iloc[0]

if pd.isna(main_camera_number):
    raise ValueError("Enter a valid main rear camera value.")

camera_1 = float(main_camera_number)

# Ignore the second rear camera.
camera_2 = 0.0


# --------------------------------------------------
# PROCESSOR TYPE
# --------------------------------------------------

processor_types = [
    "Snapdragon",
    "MediaTek",
    "Dimensity",
    "Helio",
    "Kirin",
    "Exynos",
    "Bionic",
    "Tensor",
    "Unisoc",
    "Other"
]

print("\nSelect Processor Type:")

for i, processor_option in enumerate(processor_types, 1):
    print(f"{i}. {processor_option}")

processor_choice = int(input("Enter choice: "))

if not 1 <= processor_choice <= len(processor_types):
    raise ValueError("Invalid processor choice.")

processor_type = processor_types[processor_choice - 1]

# Retaining the existing training feature behavior.
processor = processor_type


# --------------------------------------------------
# OTHER DETAILS
# --------------------------------------------------

battery_capacity = int(
    input("Battery Capacity (mAh): ")
)

screen_size = float(
    input("Screen Size (inches): ")
)

year_input = input(
    "Launched Year [Enter if unknown]: "
).strip()

launched_year = int(year_input) if year_input else 0


# --------------------------------------------------
# VARIANT
# --------------------------------------------------

variants = [
    "Pro Max",
    "Pro",
    "Ultra",
    "Plus",
    "Fold",
    "Flip",
    "Mini",
    "Max",
    "5G"
]

variant = "Other"

for v in variants:
    if v.lower() in model_name.lower():
        variant = v
        break


# --------------------------------------------------
# CATEGORICAL FEATURES
# --------------------------------------------------

categorical_data = pd.DataFrame({
    "Company Name": [brand],
    "Processor": [processor],
    "Variant": [variant],
    "Processor Type": [processor_type]
})

x_encoded = encoder.transform(categorical_data)


# --------------------------------------------------
# NUMERIC FEATURES
# Order must match the existing dev.py model.
# Camera 2 is always 0.
# --------------------------------------------------

numeric_data = np.array([[
    mobile_weight,
    ram,
    front_camera,
    battery_capacity,
    screen_size,
    launched_year,
    rom,
    camera_1,
    camera_2
]], dtype=float)


# --------------------------------------------------
# COMBINE AND PREDICT
# --------------------------------------------------

x_final = np.hstack([
    numeric_data,
    x_encoded
])

prediction = model.predict(x_final)[0]

india_price = prediction[0]
usa_price = prediction[1]
dubai_price = prediction[2]


# --------------------------------------------------
# RESULT
# --------------------------------------------------

print("\n==============================")
print("       PRICE PREDICTION")
print("==============================")

print(f"India : ₹{india_price:,.2f}")
print(f"USA   : ${usa_price:,.2f}")
print(f"Dubai : AED {dubai_price:,.2f}")

print("==============================")