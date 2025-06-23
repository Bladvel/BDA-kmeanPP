import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Remove outliers
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

# Load the dataset
df = pd.read_csv(r'..\..\data\dataset_credit_cards_clean.csv', na_values='?')
df = df.drop(columns=['CUST_ID', 'TENURE'])
df = df.dropna()

print("Shape of the original dataset:")
print(df.shape)

cols = df.columns
df_cleaned = remove_outliers_iqr(df, cols)

print("Shape of the dataset after removing outliers:")
print(df_cleaned.shape)

# Standardize Z-scores
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Save the cleaned and scaled data to a new CSV file
df_scaled.to_csv(r'..\..\data\data_without_outliers\dataset_credit_cards_wo_scaled.csv', index=False)