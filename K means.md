
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Create dataset
X, _ = make_blobs(
    n_samples=250,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

print("Dataset (first 10 rows):")
print(X[:10])

# 2. Create and train K-Means
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

# 3. Display results
print("\nK-MEANS CLUSTERING")
print("Number of clusters:", model.n_clusters)
print("Inertia:", round(model.inertia_, 3))
print("Silhouette Score:", round(silhouette_score(X, labels), 4))

print("\nCluster Centroids:")
print(model.cluster_centers_)

print("\nSamples per cluster:")
print(np.bincount(labels))

# 4. Plot clusters
plt.figure(figsize=(9, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels, cmap="viridis", s=45, alpha=0.8
)
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X", s=250, c="red",
    edgecolors="black", label="Centroids"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering (K=3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

<img width="603" height="460" alt="image" src="https://github.com/user-attachments/assets/b4794f48-bad0-402b-92b9-ba0d031ffb31" />
<img width="897" height="671" alt="image" src="https://github.com/user-attachments/assets/1a7d5792-148a-4ff5-8777-3427b20233db" />

