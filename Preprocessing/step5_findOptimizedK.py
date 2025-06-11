import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv(r'..\data\dataset_credit_cards_clean_scaled_reduced.csv')


sse = []
k_range = range(1, 11)

#Using the elbow method to find the optimal number of clusters (k)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)  # inertia_ = SSE

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Suma de errores al cuadrado (SSE)")
plt.title("Método del codo para determinar k óptimo")
plt.grid(True)
plt.show()