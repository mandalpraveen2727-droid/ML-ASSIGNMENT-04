from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

file_path = Path(__file__).parent / "Housing.csv"

df = pd.read_csv(file_path)

print("First Five Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nMissing Values:")
print(df.isnull().sum())
feature_names = ["area","bedrooms","bathrooms","stories","parking"]

X = df[feature_names]
y = df["price"]
print("\nSelected Features:")
print(feature_names)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Coefficients:")
print("-" * 35)
for feature, coefficient in zip(feature_names,model.coef_):
    print(f"{feature:<12}: {coefficient:.2f}")

print(f"\nIntercept: {model.intercept_:.2f}")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

comparison = pd.DataFrame({"Actual Price": y_test.values,"Predicted Price": y_pred})

comparison["Difference"] = (comparison["Actual Price"]- comparison["Predicted Price"])

print("\nActual Price vs Predicted Price:")
print(comparison.head(10).round(2))

print("\nEnter New House Information")
print("-" * 35)
area = float(input("Enter house area in square feet: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))
parking = int(input("Enter number of parking spaces: "))

new_house = pd.DataFrame({"area": [area],"bedrooms": [bedrooms],"bathrooms": [bathrooms],"stories": [stories],"parking": [parking]})

predicted_price = model.predict(new_house)[0]
print("\nNew House Information:")
print(new_house)
print(f"\nPredicted House Price: "f"{predicted_price:.2f}")
