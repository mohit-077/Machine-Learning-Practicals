
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score

# 1. Create dataset
X, _ = make_blobs(
    n_samples=60,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

print("Dataset (first 10 rows):")
print(X[:10])

# 2. Perform hierarchical clustering
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)
labels = model.fit_predict(X)

# 3. Display results
print("\nHIERARCHICAL CLUSTERING")
print("Number of clusters:", len(np.unique(labels)))
print("Silhouette Score:", round(silhouette_score(X, labels), 4))
print("\nSamples per cluster:")
print(np.bincount(labels))

# 4. Plot final clusters
plt.figure(figsize=(9, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels, cmap="viridis", s=55
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Agglomerative Hierarchical Clustering")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Plot dendrogram
Z = linkage(X, method="ward")

plt.figure(figsize=(13, 6))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=25,
    leaf_rotation=90,
    leaf_font_size=9
)
plt.xlabel("Samples / Merged Clusters")
plt.ylabel("Ward Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.tight_layout()
plt.show()

<img width="724" height="366" alt="image" src="https://github.com/user-attachments/assets/21401aa5-b9ce-4ddd-9af0-ee2c6015043e" />
<img width="900" height="672" alt="image" src="https://github.com/user-attachments/assets/d9742790-db25-40a4-b3fe-86845ec74690" />
<img width="1295" height="670" alt="image" src="https://github.com/user-attachments/assets/db9c9a9c-cf3b-41c3-a2f4-4ff8d13e9624" />


