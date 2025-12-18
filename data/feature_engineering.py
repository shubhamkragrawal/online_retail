"""
Feature Engineering for Customer Churn Classification
Loads data from normalized database and creates ML features
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ChurnFeatureEngineer:
    def __init__(self, db_path='database/retail.db'):
        self.db_path = db_path
        
    def load_from_database(self):
        """Load data from normalized database"""
        print("Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        
        # Load transaction data using the view
        query = """
        SELECT 
            CustomerID,
            InvoiceNo,
            InvoiceDate,
            Country,
            StockCode,
            Description,
            Quantity,
            UnitPrice,
            TotalPrice
        FROM TransactionView
        ORDER BY InvoiceDate
        """
        
        df = pd.read_sql_query(query, conn)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        conn.close()
        print(f"✓ Loaded {len(df)} transactions for {df['CustomerID'].nunique()} customers")
        return df
        
    def calculate_rfm_features(self, df, reference_date=None):
        """Calculate RFM (Recency, Frequency, Monetary) features"""
        print("Calculating RFM features...")
        
        if reference_date is None:
            reference_date = df['InvoiceDate'].max() + timedelta(days=1)
        
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        return rfm
        
    def calculate_customer_features(self, df):
        """Calculate additional customer behavior features"""
        print("Calculating customer features...")
        
        features = df.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',  # Total number of orders
            'InvoiceDate': ['min', 'max'],  # First and last purchase
            'Quantity': ['sum', 'mean'],  # Total and average items per transaction
            'TotalPrice': ['sum', 'mean', 'std'],  # Monetary metrics
            'StockCode': 'nunique',  # Product variety
            'Country': 'first'
        })
        
        # Flatten column names
        features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                           for col in features.columns.values]
        features.reset_index(inplace=True)
        
        # Calculate customer lifetime (in days)
        features['CustomerLifetime'] = (
            pd.to_datetime(features['InvoiceDate_max']) - 
            pd.to_datetime(features['InvoiceDate_min'])
        ).dt.days
        
        # Calculate average time between purchases
        features['AvgDaysBetweenPurchases'] = (
            features['CustomerLifetime'] / features['InvoiceNo_nunique']
        )
        
        # Handle division by zero
        features['AvgDaysBetweenPurchases'].fillna(0, inplace=True)
        
        # Drop date columns
        features.drop(['InvoiceDate_min', 'InvoiceDate_max'], axis=1, inplace=True)
        
        return features
        
    def create_churn_labels(self, df, observation_period_days=90, 
                           churn_period_days=90):
        """
        Create churn labels based on customer activity
        
        Parameters:
        - observation_period_days: Period to calculate features (last N days)
        - churn_period_days: Period to check if customer returned
        
        Returns: CustomerID with churn label (1 = churned, 0 = retained)
        """
        print(f"Creating churn labels (observation: {observation_period_days}d, churn: {churn_period_days}d)...")
        
        max_date = df['InvoiceDate'].max()
        observation_end = max_date - timedelta(days=churn_period_days)
        observation_start = observation_end - timedelta(days=observation_period_days)
        
        # Customers who made purchases during observation period
        observation_df = df[
            (df['InvoiceDate'] >= observation_start) & 
            (df['InvoiceDate'] < observation_end)
        ]
        observation_customers = set(observation_df['CustomerID'].unique())
        
        # Customers who made purchases during churn period
        churn_df = df[df['InvoiceDate'] >= observation_end]
        active_customers = set(churn_df['CustomerID'].unique())
        
        # Create labels
        churned_customers = observation_customers - active_customers
        
        labels = pd.DataFrame({
            'CustomerID': list(observation_customers),
            'Churned': [1 if c in churned_customers else 0 for c in observation_customers]
        })
        
        churn_rate = labels['Churned'].mean()
        print(f"✓ Churn rate: {churn_rate:.2%} ({labels['Churned'].sum()}/{len(labels)} customers)")
        
        return labels
        
    def prepare_ml_dataset(self, observation_period_days=180, churn_period_days=90):
        """Main function to prepare complete ML dataset"""
        print("\n" + "="*60)
        print("PREPARING ML DATASET")
        print("="*60 + "\n")
        
        # Load raw data
        df = self.load_from_database()
        
        # Create churn labels
        labels = self.create_churn_labels(df, observation_period_days, churn_period_days)
        
        # Calculate observation end date for feature calculation
        max_date = df['InvoiceDate'].max()
        observation_end = max_date - timedelta(days=churn_period_days)
        
        # Use only data from observation period for features
        df_obs = df[df['InvoiceDate'] < observation_end]
        
        # Calculate RFM features
        rfm = self.calculate_rfm_features(df_obs, observation_end)
        
        # Calculate additional features
        customer_features = self.calculate_customer_features(df_obs)
        
        # Merge all features
        ml_df = labels.merge(rfm, on='CustomerID')
        ml_df = ml_df.merge(customer_features, on='CustomerID')
        
        # Encode country (one-hot encoding for top countries)
        top_countries = ml_df['Country_first'].value_counts().head(10).index
        for country in top_countries:
            ml_df[f'Country_{country}'] = (ml_df['Country_first'] == country).astype(int)
        
        # Drop original country column
        ml_df.drop('Country_first', axis=1, inplace=True)
        
        # Handle missing values
        ml_df.fillna(0, inplace=True)
        
        # Handle infinity values
        ml_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        print(f"\n✓ Final dataset: {len(ml_df)} customers, {len(ml_df.columns)-2} features")
        print(f"✓ Features: {list(ml_df.columns[2:])}")
        
        return ml_df
        
    def save_dataset(self, df, output_path='data/processed/ml_dataset.csv'):
        """Save processed dataset"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Dataset saved to: {output_path}")
        
def main():
    """Main execution function"""
    engineer = ChurnFeatureEngineer()
    
    # Prepare ML dataset
    ml_df = engineer.prepare_ml_dataset(
        observation_period_days=180,  # Use 6 months of data for features
        churn_period_days=90  # Check if customer returned in next 3 months
    )
    
    # Save dataset
    engineer.save_dataset(ml_df)
    
    # Print dataset info
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(ml_df.head())
    print("\nTarget distribution:")
    print(ml_df['Churned'].value_counts())
    print(f"\nChurn rate: {ml_df['Churned'].mean():.2%}")
    
if __name__ == "__main__":
    main()