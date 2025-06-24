import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"..\data\data_with_outliers\dataset_credit_cards_clean_with_clusters.csv")

# Plot the distribution of each numeric column by cluster

# Exclude the 'Cluster' column from the numeric columns list
numeric_cols = [col for col in df.columns if col != 'Cluster']

for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=col, data=df)
    plt.title(f"Distribución de '{col}' por Cluster")
    plt.grid(True)
    plt.show()