import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Create dataset
X, y = make_moons(n_samples=300, noise=0.20, random_state=42)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create and train ANN
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
        early_stopping=True
    )
)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("ARTIFICIAL NEURAL NETWORK")
print("Architecture: 2 input features, 16 and 8 hidden neurons, 1 binary output")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Plot decision regions
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            cmap="coolwarm", edgecolors="black")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("ANN Classification - Decision Regions")
plt.tight_layout()
plt.show()

# 7. Plot training loss
plt.figure(figsize=(9, 5))
plt.plot(model[-1].loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("ANN Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("ANN - Confusion Matrix")
plt.tight_layout()
plt.show()

<img width="544" height="359" alt="image" src="https://github.com/user-attachments/assets/7bf6f825-3070-49dc-8e51-13590cec933a" />
<img width="897" height="667" alt="image" src="https://github.com/user-attachments/assets/f6909b94-ea73-4950-8aa4-2fb2fb8af337" />
<img width="901" height="573" alt="image" src="https://github.com/user-attachments/assets/6a6866e5-059d-47ad-b395-ab51d0e601e2" />
<img width="638" height="550" alt="image" src="https://github.com/user-attachments/assets/12d5938c-56fa-46f6-a117-143ee17c367d" />



