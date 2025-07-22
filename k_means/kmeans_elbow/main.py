import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)
inertia_values = []

plt.scatter(X[:, 0], X[:, 1], color='LightBlue', s=15)
plt.title("Исходные данные")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.grid(True)
plt.show()

for k in range(1, 11):
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_model.fit(X)

    inertia = kmeans_model.inertia_
    inertia_values.append(inertia)

plt.plot(range(1, 11), inertia_values, marker='o')
plt.title("Метод локтя")
plt.xlabel("Количество кластеров K")
plt.ylabel("Инерция")
plt.grid(True)
plt.show()

final_k = 4
final_model = KMeans(n_clusters=final_k, random_state=42, n_init=10)
final_model.fit(X)

final_labels = final_model.predict(X)
final_centers = final_model.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', s=15)
plt.scatter(final_centers[:, 0], final_centers[:, 1],
            c='black', s=150, marker='X', label='Центры кластеров')
plt.title("K-Means с K=4")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.grid(True)
plt.show()