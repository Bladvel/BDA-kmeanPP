import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'..\data\dataset_credit_cards_clean.csv', na_values='?')

# Eliminate columns that are not needed for analysis
df = df.drop(columns=['CUST_ID', 'TENURE'])

# See how many rows and columns are in the dataset
# print("Shape of the dataset:")
# print(df.shape)

# Check for missing values in the dataset
# missing_values = df.isnull().sum()
# print("Missing values in each column:")
# print(missing_values)

# Drop rows with missing values
df = df.dropna()

# Check the shape of the dataset after dropping missing values
# print("Shape of the dataset after dropping missing values:")
# print(df.shape)

# Standardize Z-scores
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Save the cleaned and scaled data to a new CSV file
df_scaled.to_csv(r'..\data\dataset_credit_cards_clean_scaled.csv', index=False)
