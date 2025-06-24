import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


df_scaled = pd.read_csv(r"..\data\data_with_outliers\dataset_credit_cards_clean_scaled.csv")

N = 30
silhouettes = []

for i in range(N):
    kmeans = KMeans(n_clusters=6, init='random', random_state=i)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    silhouettes.append(score)

print("Resultados de estabilidad con K-means (init='random'):")
print(f"Media Silhouette: {np.mean(silhouettes):.4f}")
print(f"Desviación estándar: {np.std(silhouettes):.4f}")


silhouettes = []

for i in range(N):
    kmeans = KMeans(n_clusters=6, init='k-means++', random_state=i)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    silhouettes.append(score)

print("Resultados de estabilidad con K-means++:")
print(f"Media Silhouette: {np.mean(silhouettes):.4f}")
print(f"Desviación estándar: {np.std(silhouettes):.4f}")