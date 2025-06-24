import pandas as pd
from sklearn.cluster import KMeans


df = pd.read_csv(r"..\data\dataset_credit_cards_clean.csv", na_values='?')
df = df.drop(columns=['CUST_ID', 'TENURE', 'PURCHASES'])
df = df.dropna()


labels = KMeans(n_clusters=6, init='k-means++', random_state=42).fit_predict(df)
df['Cluster'] = labels

# Save the DataFrame with clusters to a new CSV file
df.to_csv(r"..\data\data_with_outliers\dataset_credit_cards_clean_with_clusters.csv", index=False)

# df = pd.read_csv(r"..\data\data_with_outliers\dataset_credit_cards_clean_with_clusters.csv")

means = df.groupby('Cluster').mean(numeric_only=True).round(2)

# Save the means of each cluster to a new CSV file
means.to_csv(r"..\data\data_with_outliers\dataset_credit_cards_clean_with_clusters_means.csv", index=False)


