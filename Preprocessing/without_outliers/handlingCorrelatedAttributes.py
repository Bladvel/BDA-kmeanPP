import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'..\..\data\data_without_outliers\dataset_credit_cards_wo_scaled.csv')

plt.figure(figsize=(14, 14))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matriz de correlación entre variables (datos estandarizados)")
plt.tight_layout()
plt.show()

# Eliminate correlated attributes
df_reduced = df.drop(columns=['PURCHASES'])
df_reduced.to_csv(r'..\..\data\data_without_outliers\dataset_credit_cards_wo_scaled_reduced.csv', index=False)