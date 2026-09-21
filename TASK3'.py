# Import required libraries
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

# 1. Load the same Housing dataset
file_path = Path(__file__).parent / "Housing.csv"
df = pd.read_csv(file_path)
print("First Five Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
# 2. Select feature and target
X = df[["area"]]
y = df["price"]
# 3. Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
# Standard Linear Regression
# 4. Create and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict testing data
linear_predictions = linear_model.predict(X_test)

# Calculate evaluation metrics
linear_mae = mean_absolute_error(y_test,linear_predictions)
linear_mse = mean_squared_error(y_test,linear_predictions)
linear_rmse = linear_mse ** 0.5
linear_r2 = r2_score(y_test,linear_predictions)

# Polynomial Regression
# 5. Create polynomial features
# Degree 2 generates area and area²
polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)
# Fit and transform the training data
X_train_polynomial = (polynomial_converter.fit_transform(X_train))

# Transform testing data using the fitted converter
X_test_polynomial = (polynomial_converter.transform(X_test))
print("\nPolynomial Feature Names:")
print(polynomial_converter.get_feature_names_out(["area"]))

# 6. Create and train the Polynomial Regression model
polynomial_model = LinearRegression()
polynomial_model.fit(X_train_polynomial,y_train)
# Predict testing data
polynomial_predictions = polynomial_model.predict(X_test_polynomial)
# Calculate evaluation metrics
polynomial_mae = mean_absolute_error( y_test, polynomial_predictions)
polynomial_mse = mean_squared_error(y_test,polynomial_predictions)
polynomial_rmse = polynomial_mse ** 0.5
polynomial_r2 = r2_score(y_test,polynomial_predictions)

# 7. Compare the models
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

# 8. Predict price for a user-provided area
house_area = float(input("\nEnter house area in square feet: "))
new_house = pd.DataFrame({"area": [house_area]})

# Standard Linear Regression prediction
linear_price = linear_model.predict(new_house)[0]

# Convert the new value into polynomial features
new_house_polynomial = polynomial_converter.transform(new_house)

# Polynomial Regression prediction
polynomial_price = polynomial_model.predict(new_house_polynomial)[0]

print("\nPrediction Results:")
print(f"Linear Regression Price   : {linear_price:.2f}")
print(f"Polynomial Regression Price: {polynomial_price:.2f}")
# 9. Plot both regression models
# Generate evenly spaced area values for smooth lines
area_values = np.linspace(df["area"].min(),df["area"].max(),500).reshape(-1, 1)
# Convert to DataFrame to preserve the feature name
area_dataframe = pd.DataFrame(area_values,columns=["area"])

# Linear Regression line
linear_curve = linear_model.predict(area_dataframe)

# Polynomial Regression curve
polynomial_area = polynomial_converter.transform(area_dataframe)
polynomial_curve = polynomial_model.predict(polynomial_area)

# Draw the graph
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