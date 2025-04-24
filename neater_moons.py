from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, KMeans
from DBCV import DBCV as dbbcv
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean

n_samples=150
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
X = noisy_moons[0]

hdbscanner = HDBSCAN()
hdbscan_labels = hdbscanner.fit_predict(X)
hdbscan_dbcv_score = dbbcv(X, hdbscan_labels, dist_function=euclidean)

kmeansscanner = KMeans(n_clusters=2)
kmeans_labels = kmeansscanner.fit_predict(X)
kmeans_dbcv_score = dbbcv(X, kmeans_labels, dist_function=euclidean)


hdbscan_silhouette_score = silhouette_score(X, hdbscan_labels)
kmeans_silhouette_score = silhouette_score(X, kmeans_labels)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# HDBSCAN plot
axes[0].scatter(X[:, 0], X[:, 1], c=hdbscan_labels)
axes[0].set_title(f"HDBSCAN Clustering\nDBCV Score: {hdbscan_dbcv_score:.2f}\nSilhouette Score: {hdbscan_silhouette_score:.2f}")

# KMeans plot
axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels)
axes[1].set_title(f"KMeans Clustering\nDBCV Score: {kmeans_dbcv_score:.2f}\nSilhouette Score: {kmeans_silhouette_score:.2f}")

plt.tight_layout()
plt.show()



