from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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
X = df[["area"]]

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

slope = model.coef_[0]
intercept = model.intercept_

print("\nLinear Regression Model:")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

print("\nRegression Equation:")
print(f"Price = {intercept:.2f} + ({slope:.2f} × Area)")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

house_area = float(input("\nEnter house area in square feet: "))

new_house = pd.DataFrame({
    "area": [house_area]})
predicted_price = model.predict(new_house)[0]

print(f"\nPredicted price for a {house_area:.2f} sq. ft. house: "f"{predicted_price:.2f}")
sorted_df = df.sort_values(by="area")
sorted_area = sorted_df[["area"]]

regression_line = model.predict(sorted_area)
plt.figure(figsize=(9, 6))
plt.scatter(df["area"],df["price"],color="blue",alpha=0.5,label="Actual House Prices")
plt.plot(sorted_area["area"],regression_line,color="red",linewidth=2,label="Regression Line")
plt.scatter(house_area,predicted_price,color="green",s=120,marker="X",label="Predicted House")

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("House Price Prediction Based Only on Area")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
