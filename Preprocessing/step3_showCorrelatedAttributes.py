import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'..\data\dataset_credit_cards_clean_scaled.csv')

plt.figure(figsize=(14, 14))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matriz de correlación entre variables (datos estandarizados)")
plt.tight_layout()
plt.show()