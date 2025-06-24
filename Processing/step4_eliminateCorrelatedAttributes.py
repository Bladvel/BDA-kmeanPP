import pandas as pd

df = pd.read_csv(r'..\data\dataset_credit_cards_clean_scaled.csv')
df_reduced = df.drop(columns=['PURCHASES'])
df_reduced.to_csv(r'..\data\dataset_credit_cards_clean_scaled_reduced.csv', index=False)