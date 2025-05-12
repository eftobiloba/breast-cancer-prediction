import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # True labels (for adjusted Rand index)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the clustering algorithms
kmeans = KMeans(n_clusters=2, random_state=42)
agg_clustering = AgglomerativeClustering(n_clusters=2)

# Fit the clustering algorithms to the scaled data
kmeans_labels = kmeans.fit_predict(X_scaled)
agg_labels = agg_clustering.fit_predict(X_scaled)

# Evaluate K-means clustering performance
metrics_kmeans = {
    'Silhouette Score': silhouette_score(X_scaled, kmeans_labels),
    'Davies-Bouldin Index': davies_bouldin_score(X_scaled, kmeans_labels),
    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, kmeans_labels),
    'Adjusted Rand Index': adjusted_rand_score(y, kmeans_labels)
}

print("K-means Clustering Performance:")
for metric_name, metric_value in metrics_kmeans.items():
    print(f"{metric_name}: {metric_value:.4f}")

# Evaluate Agglomerative Clustering performance
metrics_agg = {
    'Silhouette Score': silhouette_score(X_scaled, agg_labels),
    'Davies-Bouldin Index': davies_bouldin_score(X_scaled, agg_labels),
    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, agg_labels),
    'Adjusted Rand Index': adjusted_rand_score(y, agg_labels)
}

print("\nAgglomerative Clustering Performance:")
for metric_name, metric_value in metrics_agg.items():
    print(f"{metric_name}: {metric_value:.4f}")

# Elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Silhouette analysis
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal K')
plt.show()


def compute_gap_statistic(data, k_range, n_ref_samples=10, random_seed=None):
    np.random.seed(random_seed)
    gap_stats = []
    gap_stds = []

    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, random_state=random_seed)
        kmeans_model.fit(data)
        Wk = np.log(kmeans_model.inertia_)

        ref_Wks = []
        for _ in range(n_ref_samples):
            ref_data = np.random.rand(*data.shape)
            ref_kmeans_model = KMeans(n_clusters=k, random_state=random_seed)
            ref_kmeans_model.fit(ref_data)
            ref_Wks.append(np.log(ref_kmeans_model.inertia_))

        gap_stat = np.mean(ref_Wks) - Wk
        gap_std = np.std(ref_Wks) * np.sqrt(1 + 1/n_ref_samples)
        gap_stats.append(gap_stat)
        gap_stds.append(gap_std)

    return np.array(gap_stats), np.array(gap_stds)

# Gap Statistic
k_range = range(1, 11)
gap_stats, gap_stds = compute_gap_statistic(X_scaled, k_range, n_ref_samples=10, random_seed=42)

plt.figure(figsize=(10, 6))
plt.plot(k_range, gap_stats, marker='o', color='b', label='Gap Statistic')
plt.errorbar(k_range, gap_stats, yerr=gap_stds, fmt='-o', color='b', alpha=0.5, label='Gap Statistic with Std Dev')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic for Optimal k')
plt.xticks(k_range)
plt.legend()
plt.grid(True)
plt.show()

# Dendrogram for Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
