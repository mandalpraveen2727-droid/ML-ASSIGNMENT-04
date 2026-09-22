from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

file_path = Path(__file__).parent / "Housing.csv"
df = pd.read_csv(file_path)
print("First Five Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
X = df[["area"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test,linear_predictions)
linear_mse = mean_squared_error(y_test,linear_predictions)
linear_rmse = linear_mse ** 0.5
linear_r2 = r2_score(y_test,linear_predictions)

polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)
X_train_polynomial = (polynomial_converter.fit_transform(X_train))

X_test_polynomial = (polynomial_converter.transform(X_test))
print("\nPolynomial Feature Names:")
print(polynomial_converter.get_feature_names_out(["area"]))

polynomial_model = LinearRegression()
polynomial_model.fit(X_train_polynomial,y_train)
polynomial_predictions = polynomial_model.predict(X_test_polynomial)
polynomial_mae = mean_absolute_error( y_test, polynomial_predictions)
polynomial_mse = mean_squared_error(y_test,polynomial_predictions)
polynomial_rmse = polynomial_mse ** 0.5
polynomial_r2 = r2_score(y_test,polynomial_predictions)

print("\nModel Performance Comparison")
print("-" * 65)
print(f"{'Model':<25}"f"{'MAE':>12}"f"{'RMSE':>12}"f"{'R² Score':>12}")
print("-" * 65)
print(f"{'Linear Regression':<25}"f"{linear_mae:>12.2f}"f"{linear_rmse:>12.2f}"f"{linear_r2:>12.4f}")
print(f"{'Polynomial Regression':<25}"f"{polynomial_mae:>12.2f}"f"{polynomial_rmse:>12.2f}"f"{polynomial_r2:>12.4f}")

print("-" * 65)

if polynomial_r2 > linear_r2:
    print("\nPolynomial Regression performs better ""because it has a higher R² score.")
elif linear_r2 > polynomial_r2:
    print("\nStandard Linear Regression performs better ""because it has a higher R² score.")
else:
    print("\nBoth models have the same R² score.")

house_area = float(input("\nEnter house area in square feet: "))
new_house = pd.DataFrame({"area": [house_area]})

linear_price = linear_model.predict(new_house)[0]

new_house_polynomial = polynomial_converter.transform(new_house)

polynomial_price = polynomial_model.predict(new_house_polynomial)[0]

print("\nPrediction Results:")
print(f"Linear Regression Price   : {linear_price:.2f}")
print(f"Polynomial Regression Price: {polynomial_price:.2f}")

area_values = np.linspace(df["area"].min(),df["area"].max(),500).reshape(-1, 1)
area_dataframe = pd.DataFrame(area_values,columns=["area"])

linear_curve = linear_model.predict(area_dataframe)

polynomial_area = polynomial_converter.transform(area_dataframe)
polynomial_curve = polynomial_model.predict(polynomial_area)

plt.figure(figsize=(10, 6))
plt.scatter(df["area"],df["price"],color="blue",alpha=0.4,label="Actual House Prices")

plt.plot(area_values,linear_curve,color="red",linewidth=2,
    label=f"Linear Regression (R² = {linear_r2:.4f})")

plt.plot(area_values,polynomial_curve,color="green",linewidth=2,
    label=f"Polynomial Regression (R² = {polynomial_r2:.4f})")
plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("Linear Regression vs Polynomial Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
