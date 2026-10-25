import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Create dataset
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, class_sep=1.2, random_state=42
)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train model
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("LOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Plot decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            cmap="coolwarm", edgecolors="black")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Regions")
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()

<img width="612" height="322" alt="image" src="https://github.com/user-attachments/assets/2d06367c-31a8-4f35-b2d0-e493f5b4f488" />
<img width="905" height="671" alt="image" src="https://github.com/user-attachments/assets/828ff3be-3486-476e-9d60-982cbfa3de81" />
<img width="633" height="552" alt="image" src="https://github.com/user-attachments/assets/6de7a6dd-805c-4814-8918-ac98a6322100" />


