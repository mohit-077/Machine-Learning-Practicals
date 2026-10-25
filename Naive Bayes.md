
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Create dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

print("Dataset (first 10 rows):")
print("Feature 1 | Feature 2 | Class")
for i in range(10):
    print(f"{X[i, 0]:9.3f} | {X[i, 1]:9.3f} | {y[i]}")

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nNaive Bayes Classifier")
print("Accuracy:", round(accuracy, 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Plot decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_test, cmap="coolwarm",
    edgecolors="black", s=55, label="Test samples"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Gaussian Naive Bayes Classifier")
plt.legend()
plt.tight_layout()

# Show figure in a separate window
plt.show()

# 7. Confusion matrix figure
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()
plt.show()

<img width="553" height="398" alt="image" src="https://github.com/user-attachments/assets/e17c337b-ebce-4f9c-9a7f-d930e699c8a2" />
<img width="1002" height="667" alt="image" src="https://github.com/user-attachments/assets/75d9d67b-28ba-40ae-8d62-553dda675b64" />


