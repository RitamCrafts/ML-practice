import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv("Mobiles Dataset (2025).csv")
print(df)
print(df.head())
print(df.columns)
print(df.shape)
print(df.dtypes)
print("----------------------------------------")
y = df[["Launched Price (India)", "Launched Price (USA)", "Launched Price (Dubai)"]]
x = df.drop(columns=["Launched Price (India)", "Launched Price (USA)", "Launched Price (Dubai)"])


print("------------------DATA CLEAN x----------------------")
#print(x["Model Name"])
rom = x["Model Name"].str.extract(
    r"(\d+(?:\.\d+)?)\s*(TB|GB)\b",
    expand=False
)
x["ROM"] = rom[0].astype(float)
x.loc[rom[1] == "TB", "ROM"] *= 1024
# print(x["ROM"].isna().sum())
# print(x.loc[x["ROM"].isna(), "Model Name"].to_list())
x["ROM"] = x["ROM"].fillna(x["ROM"].median())
x["Model"] = x["Model Name"].str.rsplit(" ", n=1).str[0]
x=x.drop(columns=["Model Name"])
# print(x[["ROM","Model","Pro"]])
# print(x["ROM"].unique())


# print(x[["Model"]])
# print(x["Model"].nunique())
# print(x["Model"].value_counts().head(100))
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
x["Variant"] = "Other"
for i in variants:
    mask = x["Model"].str.contains(i, case=False, na=False)
    x.loc[mask, "Variant"] = i
x=x.drop(columns=["Model"])
#print(x["Variant"])
#so for future ref we got Variant need hot enc and we got ROM from this


#print(x["Mobile Weight"].head())
x["Mobile Weight"]=x["Mobile Weight"].str.replace("g","").astype(float)
#print(x["Mobile Weight"].head())

# print(x["RAM"].unique())
# print(x["RAM"].head())
x["RAM"] = x["RAM"].str.extract(r"([\d.]+)").astype(float) #regex from AI
# print(x["RAM"].head())

# print(x["Front Camera"].unique())
# print(x["Front Camera"].head())
x["Front Camera"] = x["Front Camera"].str.extract(r"([\d.]+)").astype(float) #regex from AI
# print(x["Front Camera"].head())

# print(x["Back Camera"].unique())
# print(x["Back Camera"].head())
camera_cols = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]
x[camera_cols]=(x["Back Camera"].str.split("+", expand=True))
for i in camera_cols:
    x[i] = x[i].str.extract(r"([\d.]+)").astype(float)
x[camera_cols] = x[camera_cols].fillna(0)
x=x.drop(columns=["Back Camera"])
# print(x[camera_cols].head())
# print(x[camera_cols].dtypes)


# print(x["Processor"].nunique())
# print(x["Processor"].value_counts().head(100))
processor_types = [
    "Snapdragon",
    "MediaTek",
    "Dimensity",
    "Helio",
    "Kirin",
    "Exynos",
    "Bionic",
    "Tensor",
    "Unisoc"
]
x["Processor Type"] = "Other"
for processor in processor_types:
    mask = x["Processor"].str.contains(processor, case=False, na=False)
    x.loc[mask, "Processor Type"] = processor
#print(x["Processor Type"].value_counts())


x["Battery Capacity"] = x["Battery Capacity"].str.extract(r"([\d.]+)").astype(int) #regex from AI
x["Screen Size"] = x["Screen Size"].str.extract(r"([\d.]+)").astype(float) #regex from AI

print(x.dtypes)
print(x.head())


print("------------------DATA CLEAN y----------------------")
for col in y.columns:
    y[col] = y[col].str.replace(",", "")
    y[col] = y[col].str.extract(r"([\d.]+)").astype(float)
print(y.dtypes)
print(y.head())



print("----------------------TRAIN TEST SPLIT------------------------------------")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print("-----------------------HOT ENC x--------------------------")
categorical_cols = [
    "Company Name",
    "Processor",
    "Variant",
    "Processor Type"
]
encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
encoder.fit(x_train[categorical_cols])
joblib.dump(encoder,"Mobiles-Dataset-Encoder.pkl")
x_train_encoded = encoder.transform(x_train[categorical_cols])
x_test_encoded = encoder.transform(x_test[categorical_cols])

numeric_cols = []
for col in x.columns:
    if col not in categorical_cols:
        numeric_cols.append(col)

x_train_numeric = x_train[numeric_cols].to_numpy()
x_test_numeric = x_test[numeric_cols].to_numpy()

x_train_final = np.hstack([x_train_numeric,x_train_encoded])
x_test_final = np.hstack([x_test_numeric,x_test_encoded])

print(x_train_final.shape)
print(x_test_final.shape)
print(x_train_final[:5])

print("---------------------MODEL FITTING FINALLY------------------------")
#model = LinearRegression()

#model = DecisionTreeRegressor(max_depth=10, random_state=42)

model = RandomForestRegressor(n_estimators=50,max_depth=20,random_state=42)

#model = ExtraTreesRegressor(n_estimators=200,max_depth=10,random_state=42)

#model = GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=42)


model.fit(x_train_final, y_train)
y_pred_test = model.predict(x_test_final)
y_pred_test_df = pd.DataFrame(
    y_pred_test,
    columns=["Launched Price (India)", "Launched Price (USA)", "Launched Price (Dubai)"]
)
y_pred_train = model.predict(x_train_final)
y_pred_train_df = pd.DataFrame(
    y_pred_train,
    columns=["Launched Price (India)", "Launched Price (USA)", "Launched Price (Dubai)"]
)
print("---------------------RESULTS------------------------")
y_test = y_test.reset_index(drop=True)
print(y_pred_test_df["Launched Price (India)"].head())
print(y_test["Launched Price (India)"].head())

comparison = pd.DataFrame({
    "Predicted": y_pred_test_df["Launched Price (India)"].head(30),
    "Actual": y_test["Launched Price (India)"].head(30)
})
comparison["Error"] = comparison["Predicted"] - comparison["Actual"]
comparison["Absolute Error"] = abs(comparison["Error"])
print(comparison)

mae_test = mean_absolute_error(y_test["Launched Price (India)"], y_pred_test_df["Launched Price (India)"])
print("MAE test:", mae_test)
r2_test = r2_score(y_test["Launched Price (India)"], y_pred_test_df["Launched Price (India)"])
print("R squared test:", r2_test)
mae_train = mean_absolute_error(y_train["Launched Price (India)"], y_pred_train_df["Launched Price (India)"])
print("MAE train:", mae_train)
r2_train = r2_score(y_train["Launched Price (India)"], y_pred_train_df["Launched Price (India)"])
print("R squared train:", r2_train)

op = input("Would you like to save your model(y/n):").upper()
if op == 'Y':
    joblib.dump(model, "phone_price_model.pkl")