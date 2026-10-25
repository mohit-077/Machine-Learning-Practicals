
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Create dataset
data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
        "Overcast", "Sunny", "Sunny", "Rain", "Sunny",
        "Overcast", "Overcast", "Rain"
    ],
    "Temperature": [
        "Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
        "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"
    ],
    "Humidity": [
        "High", "High", "High", "High", "Normal", "Normal",
        "Normal", "High", "Normal", "Normal", "Normal",
        "High", "Normal", "High"
    ],
    "Wind": [
        "Weak", "Strong", "Weak", "Weak", "Weak", "Strong",
        "Strong", "Weak", "Weak", "Weak", "Strong",
        "Strong", "Weak", "Strong"
    ],
    "PlayTennis": [
        "No", "No", "Yes", "Yes", "Yes", "No", "Yes",
        "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"
    ]
}

df = pd.DataFrame(data)
print("Dataset:\n")
print(df)

# 2. Encode categorical features
encoders = {}
encoded = df.copy()

for col in df.columns:
    encoders[col] = LabelEncoder()
    encoded[col] = encoders[col].fit_transform(df[col])

X = encoded.drop(columns="PlayTennis")
y = encoded["PlayTennis"]

# 3. Create and train model
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

# 4. Predict
y_pred = model.predict(X)

# 5. Evaluate
print("\nDECISION TREE CLASSIFICATION")
print("Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n",
      classification_report(y, y_pred, zero_division=0))

# 6. Plot decision tree
plt.figure(figsize=(16, 9))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders["PlayTennis"].classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree - Play Tennis")
plt.tight_layout()
plt.show()

<img width="647" height="632" alt="image" src="https://github.com/user-attachments/assets/afae7694-283b-4808-8b4a-e844a6eb39d4" />
<img width="1596" height="969" alt="image" src="https://github.com/user-attachments/assets/8d1b3b1a-bb63-4bc2-87a0-e2d26431248c" />

