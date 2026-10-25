import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create dataset
X, y = make_regression(
    n_samples=200, n_features=5, n_informative=3,
    noise=20, random_state=42
)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train models
lasso = make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=10000))
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

# 4. Predict
lasso_pred = lasso.predict(X_test)
ridge_pred = ridge.predict(X_test)

# 5. Evaluate
for name, pred in [("Lasso", lasso_pred), ("Ridge", ridge_pred)]:
    print(f"\n{name} Regression")
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2 Score:", r2_score(y_test, pred))

# 6. Plot actual vs predicted
plt.figure(figsize=(9, 6))
plt.scatter(y_test, lasso_pred, label="Lasso", alpha=0.7)
plt.scatter(y_test, ridge_pred, label="Ridge", alpha=0.7)

low = min(y_test.min(), lasso_pred.min(), ridge_pred.min())
high = max(y_test.max(), lasso_pred.max(), ridge_pred.max())
plt.plot([low, high], [low, high], "--", linewidth=2)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Lasso vs Ridge Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

<img width="540" height="187" alt="image" src="https://github.com/user-attachments/assets/226e84f5-6bd3-475a-bf2f-6f16d8e7db23" />
<img width="902" height="675" alt="image" src="https://github.com/user-attachments/assets/675210a6-2541-4c9a-8e9f-83ad35ce0cc8" />




