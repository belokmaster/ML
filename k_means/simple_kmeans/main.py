import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)

plt.scatter(X[:, 0], X[:, 1], color='LightBlue', s=15)
plt.title("Исходные данные")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.grid(True)
plt.show()

kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(X)

predicted_labels = kmeans_model.labels_
cluster_centers = kmeans_model.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=15)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            c='black', s=100, marker='X', label='Центры кластеров')
plt.title("Результат кластеризации K-Means")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.grid(True)
plt.show()