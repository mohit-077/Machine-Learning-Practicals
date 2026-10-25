import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create nonlinear dataset
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 2 + 1.5 * X[:, 0] + 2 * X[:, 0]**2 + np.random.normal(0, 2, 100)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Convert features into polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 4. Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test_poly)

print("POLYNOMIAL REGRESSION")
print("Degree: 2")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 6. Plot fitted curve
X_curve = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_curve = model.predict(poly.transform(X_curve))

plt.figure(figsize=(9, 6))
plt.scatter(X, y, label="Original data")
plt.plot(X_curve, y_curve, linewidth=2, label="Polynomial curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression (Degree 2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


<img width="562" height="156" alt="image" src="https://github.com/user-attachments/assets/5211c625-2127-4c58-bfa2-ba60c5b4319c" />
<img width="897" height="672" alt="image" src="https://github.com/user-attachments/assets/66c4e93c-9207-49b1-9211-b2909adf600b" />




