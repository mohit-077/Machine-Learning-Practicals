import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Create dataset
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train k-NN
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("k-NEAREST NEIGHBOURS CLASSIFIER")
print("K =", 5)
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
plt.title("k-NN Classification (k=5)")
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("k-NN - Confusion Matrix")
plt.tight_layout()
plt.show()

<img width="615" height="330" alt="image" src="https://github.com/user-attachments/assets/fd16ab66-e5a8-459c-a442-0ea85b7afdf4" />
<img width="899" height="673" alt="image" src="https://github.com/user-attachments/assets/5d6956e9-029b-4895-ad07-6ad2b11d3bc3" />
<img width="644" height="550" alt="image" src="https://github.com/user-attachments/assets/bc01ad93-d8f8-450b-b4a7-67f38417c423" />


