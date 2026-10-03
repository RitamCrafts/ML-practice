import pandas as pd
import numpy as np
import joblib


# Load model and encoder
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

brand = brand_options[brand_choice - 1]


# --------------------------------------------------
# PHONE DETAILS
# --------------------------------------------------

model_name = input("\nModel Name: ")

rom = float(input("ROM (GB): "))


weight_input = input(
    "Mobile Weight (g) [Enter if unknown]: "
)

if weight_input == "":
    mobile_weight = 0.0
else:
    mobile_weight = float(weight_input)


ram = float(input("RAM (GB): "))


# --------------------------------------------------
# FRONT CAMERA
# --------------------------------------------------

front_camera = input(
    "Front Camera (example: 12MP): "
)

front_camera = float(
    pd.Series([front_camera])
    .str.extract(r"([\d.]+)", expand=False)
    .iloc[0]
)


# --------------------------------------------------
# BACK CAMERA
# --------------------------------------------------

back_camera = input(
    "Back Camera (example: 50MP + 12MP + 10MP): "
)


# --------------------------------------------------
# PROCESSOR TYPE SELECTION
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

for i, processor_type in enumerate(processor_types, 1):
    print(f"{i}. {processor_type}")

processor_choice = int(input("Enter choice: "))

processor_type = processor_types[processor_choice - 1]

# The model also has a raw Processor feature.
# Since the user only selects Processor Type,
# use the selected type as the processor value.
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
)

if year_input == "":
    launched_year = 0
else:
    launched_year = int(year_input)


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
# BACK CAMERA
# --------------------------------------------------

camera_parts = back_camera.split("+")

camera_values = []

for camera in camera_parts[:4]:

    number = pd.Series([camera]).str.extract(
        r"([\d.]+)",
        expand=False
    ).iloc[0]

    if pd.isna(number):
        camera_values.append(0.0)
    else:
        camera_values.append(float(number))


while len(camera_values) < 4:
    camera_values.append(0.0)


camera_1 = camera_values[0]
camera_2 = camera_values[1]
camera_3 = camera_values[2]
camera_4 = camera_values[3]


# --------------------------------------------------
# CATEGORICAL FEATURES
# --------------------------------------------------

categorical_data = pd.DataFrame({

    "Company Name": [brand],

    "Processor": [processor],

    "Variant": [variant],

    "Processor Type": [processor_type]

})


x_encoded = encoder.transform(
    categorical_data
)


# --------------------------------------------------
# NUMERIC FEATURES
# SAME ORDER AS TRAINING
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
    camera_2,
    camera_3,
    camera_4
]])


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

x_final = np.hstack([
    numeric_data,
    x_encoded
])


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

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