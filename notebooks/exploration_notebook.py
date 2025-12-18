# Data Exploration Notebook
# This is a Python script version - convert to .ipynb using: jupytext --to notebook this_file.py

# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries imported successfully!")

# Cell 2: Load Data from Database
conn = sqlite3.connect('database/retail.db')

# Load transaction data
query = """
SELECT * FROM TransactionView
LIMIT 10000
"""
df = pd.read_sql_query(query, conn)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
df.head()

# Cell 3: Basic Statistics
print("Dataset Info:")
print(df.info())

print("\nNumerical Statistics:")
df.describe()

# Cell 4: Missing Values
print("Missing Values:")
missing = df.isnull().sum()
missing[missing > 0]

# Cell 5: Customer Analysis
customer_stats = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum',
    'Quantity': 'sum',
    'InvoiceDate': ['min', 'max']
}).reset_index()

customer_stats.columns = ['CustomerID', 'Orders', 'TotalSpent', 'TotalQuantity', 'FirstPurchase', 'LastPurchase']

print("Customer Statistics:")
customer_stats.describe()

# Cell 6: Visualize Customer Distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Orders distribution
axes[0, 0].hist(customer_stats['Orders'], bins=50, edgecolor='black')
axes[0, 0].set_title('Distribution of Orders per Customer')
axes[0, 0].set_xlabel('Number of Orders')
axes[0, 0].set_ylabel('Frequency')

# Spending distribution
axes[0, 1].hist(customer_stats['TotalSpent'], bins=50, edgecolor='black')
axes[0, 1].set_title('Distribution of Total Spending')
axes[0, 1].set_xlabel('Total Spent (£)')
axes[0, 1].set_ylabel('Frequency')

# Quantity distribution
axes[1, 0].hist(customer_stats['TotalQuantity'], bins=50, edgecolor='black')
axes[1, 0].set_title('Distribution of Total Quantity')
axes[1, 0].set_xlabel('Total Quantity')
axes[1, 0].set_ylabel('Frequency')

# Boxplot for spending
axes[1, 1].boxplot(customer_stats['TotalSpent'])
axes[1, 1].set_title('Total Spending - Boxplot')
axes[1, 1].set_ylabel('Total Spent (£)')

plt.tight_layout()
plt.show()

# Cell 7: Country Analysis
country_stats = df.groupby('Country').agg({
    'CustomerID': 'nunique',
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique'
}).reset_index()
country_stats.columns = ['Country', 'Customers', 'Revenue', 'Orders']
country_stats = country_stats.sort_values('Revenue', ascending=False).head(10)

print("Top 10 Countries by Revenue:")
print(country_stats)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].barh(country_stats['Country'], country_stats['Revenue'])
axes[0].set_xlabel('Revenue (£)')
axes[0].set_title('Top 10 Countries by Revenue')

axes[1].barh(country_stats['Country'], country_stats['Customers'])
axes[1].set_xlabel('Number of Customers')
axes[1].set_title('Top 10 Countries by Customers')

plt.tight_layout()
plt.show()

# Cell 8: Time Series Analysis
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
daily_sales.columns = ['Date', 'Sales']

plt.figure(figsize=(15, 6))
plt.plot(daily_sales['Date'], daily_sales['Sales'])
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales (£)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cell 9: Product Analysis
product_stats = df.groupby('Description').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique'
}).reset_index()
product_stats.columns = ['Product', 'QuantitySold', 'Revenue', 'Orders']
product_stats = product_stats.sort_values('Revenue', ascending=False).head(20)

print("Top 20 Products by Revenue:")
print(product_stats)

# Cell 10: Load ML Dataset
ml_df = pd.read_csv('data/processed/ml_dataset.csv')

print(f"ML Dataset shape: {ml_df.shape}")
print("\nFeatures:", ml_df.columns.tolist())
print("\nTarget distribution:")
print(ml_df['Churned'].value_counts())
print(f"\nChurn rate: {ml_df['Churned'].mean():.2%}")

# Cell 11: Feature Correlations
# Select numerical features only
numerical_features = ml_df.select_dtypes(include=[np.number]).columns
correlation_matrix = ml_df[numerical_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Cell 12: RFM Analysis
rfm_features = ['Recency', 'Frequency', 'Monetary']
if all(f in ml_df.columns for f in rfm_features):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, feature in enumerate(rfm_features):
        ml_df.boxplot(column=feature, by='Churned', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Churn Status')
        axes[idx].set_xlabel('Churned')
        axes[idx].set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()

# Cell 13: Feature Distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

important_features = [
    'Recency', 'Frequency', 'Monetary',
    'InvoiceNo_nunique', 'Quantity_mean', 'TotalPrice_mean',
    'StockCode_nunique', 'CustomerLifetime', 'AvgDaysBetweenPurchases'
]

for idx, feature in enumerate(important_features):
    if feature in ml_df.columns:
        axes[idx].hist(ml_df[feature], bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Cell 14: Summary Statistics
print("\n" + "="*60)
print("DATA EXPLORATION SUMMARY")
print("="*60)
print(f"\nTotal Transactions: {len(df):,}")
print(f"Unique Customers: {df['CustomerID'].nunique():,}")
print(f"Unique Products: {df['StockCode'].nunique():,}")
print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"\nML Dataset Samples: {len(ml_df):,}")
print(f"Features: {ml_df.shape[1] - 2}")  # Excluding CustomerID and Churned
print(f"Churn Rate: {ml_df['Churned'].mean():.2%}")
print(f"Class Balance: {ml_df['Churned'].value_counts().to_dict()}")
print("\n" + "="*60)

# Cell 15: Close database connection
conn.close()
print("Database connection closed.")
print("\nExploration complete! ✅")