import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv(r'..\..\data\data_without_outliers\dataset_credit_cards_wo_scaled_reduced.csv')


sse = []
silhouette_scores = []
k_range = range(2, 13)

#Using the elbow method to find the optimal number of clusters (k)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    sse.append(kmeans.inertia_)  # inertia_ = SSE
    silhouette_avg = silhouette_score(df, cluster_labels)
    silhouette_scores.append(silhouette_avg)


plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.plot(k_range, sse, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Suma de errores al cuadrado (SSE)")
plt.title("Método del codo para determinar k óptimo")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.xlabel("Número de clusters (k)")   
plt.ylabel("Puntuación de Silhouette")
plt.title("Puntuación de Silhouette para diferentes valores de k")
plt.grid(True)
plt.show()