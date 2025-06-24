import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos escalados
df_scaled = pd.read_csv(r"..\..\data\data_without_outliers\dataset_credit_cards_wo_scaled_reduced.csv")

# Función para calcular SSB (Sum of Squares Between)
def calculate_ssb(X, labels, centroids):
    overall_mean = np.mean(X, axis=0)
    unique_labels = np.unique(labels)
    ssb = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_size = cluster_points.shape[0]
        centroid = centroids[label]
        ssb += cluster_size * np.sum((centroid - overall_mean) ** 2)
    return ssb

# K-means (init='random')
kmeans = KMeans(n_clusters=6, init='random', random_state=42)
kmeans.fit(df_scaled)
labels_kmeans = kmeans.labels_
sse_kmeans = kmeans.inertia_
ssb_kmeans = calculate_ssb(df_scaled.values, labels_kmeans, kmeans.cluster_centers_)
silhouette_kmeans = silhouette_score(df_scaled, labels_kmeans)

# K-means++ (init='k-means++')
kmeans_pp = KMeans(n_clusters=6, init='k-means++', random_state=42)
kmeans_pp.fit(df_scaled)
labels_kmeanspp = kmeans_pp.labels_
sse_kmeanspp = kmeans_pp.inertia_
ssb_kmeanspp = calculate_ssb(df_scaled.values, labels_kmeanspp, kmeans_pp.cluster_centers_)
silhouette_kmeanspp = silhouette_score(df_scaled, labels_kmeanspp)

# Mostrar resultados
print("Resultados de clustering:")
print("K-means tradicional:")
print(f"  SSE:         {sse_kmeans:.2f}")
print(f"  SSB:         {ssb_kmeans:.2f}")
print(f"  Silhouette:  {silhouette_kmeans:.4f}")
print(f"  Iteraciones: {kmeans.n_iter_}")
print()
print("K-means++:")
print(f"  SSE:         {sse_kmeanspp:.2f}")
print(f"  SSB:         {ssb_kmeanspp:.2f}")
print(f"  Silhouette:  {silhouette_kmeanspp:.4f}")
print(f"  Iteraciones: {kmeans_pp.n_iter_}")

# Visualización con PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)

df_plot = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
df_plot['KMeans'] = labels_kmeans
df_plot['KMeans++'] = labels_kmeanspp

plt.figure(figsize=(14, 5))

plt.subplot(2, 1, 1)
sns.scatterplot(data=df_plot, x='PCA1', y='PCA2', hue='KMeans', palette='Set1')
plt.title('Clusters con K-means (init=random)')

plt.subplot(2, 1, 2)
sns.scatterplot(data=df_plot, x='PCA1', y='PCA2', hue='KMeans++', palette='Set2')
plt.title('Clusters con K-means++ (init=k-means++)')

plt.tight_layout()
plt.show()
