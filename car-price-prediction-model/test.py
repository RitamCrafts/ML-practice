import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn.ensemble import ExtraTreesRegressor
import joblib



df = pd.read_csv("CAR-DETAILS.csv")
print(df)
print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum()) #this check how many null are therer   
df = df.dropna()
print("---------------------------------------------------------------")

y=df["selling_price"]
x=df.drop("selling_price",axis=1)
print(x.head())
print(y.head())
print(x.dtypes)
print(x["fuel"].unique())#hot encoding possible
print(x["seller_type"].unique())#hot encoding possible
print(x["transmission"].unique())#hot encoding possible
print(x["owner"].unique())#hot encoding possible
print(x["name"].unique())

x["brand"]=x["name"].str.split().str[0]
x["model"]=x["name"].str.split().str[1]
print(x["name"].value_counts().head(20))

print(x["brand"].unique())
print(x["brand"].nunique())
print(x["brand"].value_counts())
x = x.drop(columns=["name"])

print(x.head())
print(x.dtypes)

print("----------------------TRAIN TEST SPLIT------------------------------------")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print("---------------------ONE HOT ENCODER-----------------------------------")

encoder = OneHotEncoder(handle_unknown="ignore")
print(encoder)
categorical_columns = ["fuel","seller_type","transmission","owner","brand","model"]
encoder.fit(x_train[categorical_columns])
x_train_encoded = encoder.transform(x_train[categorical_columns])
print(x_train_encoded)
print(x_train_encoded.toarray()[:5])
print(encoder.get_feature_names_out())


x_test_encoded = encoder.transform(x_test[categorical_columns])#we use same encoder like to make same column mapping

x_test_numeric = x_test[["year","km_driven"]].to_numpy()

x_train_numeric = x_train[["year","km_driven"]].to_numpy()

x_train_final = hstack([x_train_numeric, x_train_encoded])
x_test_final = hstack([x_test_numeric, x_test_encoded])

print(x_train_final.shape)
print(x_test_final.shape)

#model = DecisionTreeRegressor(max_depth=30, random_state=42)

#model = RandomForestRegressor(n_estimators=200,max_depth=30,random_state=42)

model = ExtraTreesRegressor(n_estimators=200,max_depth=30,random_state=42)

#model = GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=42)

model.fit(x_train_final, y_train)

y_pred = model.predict(x_test_final)

print("-----------------------------------RESULTS------------------------------------")

for pred, actual in zip(y_pred[:10], y_test.head(10).to_numpy()):
    print(f"Predicted: Rs.{pred:.0f} \t| Actual: Rs.{actual}")



mae_test = mean_absolute_error(y_test, y_pred)
print("MAE test:", mae_test)

r2_test = r2_score(y_test, y_pred)
print("R squared test:", r2_test)



y_pred_train = model.predict(x_train_final)

mae_train = mean_absolute_error(y_train, y_pred_train)
print("MAE train:", mae_train)

r2_train = r2_score(y_train, y_pred_train)
print("R squared train:", r2_train)

op = input("Would you like to save your model(y/n):").upper()
if op == 'Y':
    joblib.dump(model, "car_price_model.pkl")