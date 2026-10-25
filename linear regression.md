
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create dataset
X, y = make_regression(
    n_samples=200, n_features=3, noise=15, random_state=42
)

# Simple regression uses one feature
X_simple = X[:, [0]]

# 2. Split datasets
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train models
simple_model = LinearRegression()
multiple_model = LinearRegression()

simple_model.fit(Xs_train, ys_train)
multiple_model.fit(Xm_train, ym_train)

# 4. Predictions
ys_pred = simple_model.predict(Xs_test)
ym_pred = multiple_model.predict(Xm_test)

# 5. Evaluation
print("SIMPLE LINEAR REGRESSION")
print("Slope:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)
print("MSE:", mean_squared_error(ys_test, ys_pred))
print("R2 Score:", r2_score(ys_test, ys_pred))

print("\nMULTIPLE LINEAR REGRESSION")
print("Coefficients:", multiple_model.coef_)
print("Intercept:", multiple_model.intercept_)
print("MSE:", mean_squared_error(ym_test, ym_pred))
print("R2 Score:", r2_score(ym_test, ym_pred))

# 6. Simple regression figure
x_line = np.linspace(X_simple.min(), X_simple.max(), 200).reshape(-1, 1)
y_line = simple_model.predict(x_line)

plt.figure(figsize=(9, 6))
plt.scatter(X_simple[:, 0], y, alpha=0.6, label="Dataset")
plt.plot(x_line[:, 0], y_line, linewidth=2, label="Regression line")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Multiple regression: actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(ym_test, ym_pred, alpha=0.7)
low = min(ym_test.min(), ym_pred.min())
high = max(ym_test.max(), ym_pred.max())
plt.plot([low, high], [low, high], "--", linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Multiple Linear Regression")
plt.grid(True)
plt.tight_layout()
plt.show()


<img width="549" height="215" alt="image" src="https://github.com/user-attachments/assets/9ad53e22-b2a9-48e6-9981-3ba74272a539" />
<img width="898" height="668" alt="image" src="https://github.com/user-attachments/assets/3b24c980-871e-40a6-b053-ea8a1d31618e" />
<img width="801" height="671" alt="image" src="https://github.com/user-attachments/assets/07b63b56-c7e2-4a18-8b46-f26eb6c88719" />



