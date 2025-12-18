"""
Database Initialization and Population Script
Loads Online Retail CSV data into normalized 3NF SQLite database
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
import numpy as np

class RetailDatabase:
    def __init__(self, db_path='database/retail.db', schema_path='database/schema.sql'):
        self.db_path = db_path
        self.schema_path = schema_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
    def create_database(self):
        """Create database with 3NF schema"""
        print("Creating database schema...")
        conn = sqlite3.connect(self.db_path)
        
        with open(self.schema_path, 'r') as f:
            schema_sql = f.read()
        
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
        print("✓ Database schema created successfully")
        
    def load_and_clean_data(self, csv_path='data/raw/online_retail.csv'):
        """Load and clean the raw CSV data"""
        print(f"Loading data from {csv_path}...")
        
        # Read CSV with proper encoding
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        
        print(f"Initial records: {len(df)}")
        
        # Clean data
        # Remove rows with missing CustomerID
        df = df.dropna(subset=['CustomerID'])
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        # Remove cancelled orders (InvoiceNo starting with 'C')
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # Remove rows with missing or invalid data
        df = df.dropna(subset=['Description', 'Quantity', 'UnitPrice'])
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Calculate TotalPrice
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        print(f"Cleaned records: {len(df)}")
        return df
        
    def populate_customers(self, df, conn):
        """Populate Customers table"""
        print("Populating Customers table...")
        
        customer_data = df.groupby('CustomerID').agg({
            'Country': 'first',
            'InvoiceDate': ['min', 'max', 'count'],
            'TotalPrice': 'sum'
        }).reset_index()
        
        customer_data.columns = ['CustomerID', 'Country', 'FirstPurchaseDate', 
                                 'LastPurchaseDate', 'TotalPurchases', 'TotalSpent']
        
        customer_data['FirstPurchaseDate'] = customer_data['FirstPurchaseDate'].astype(str)
        customer_data['LastPurchaseDate'] = customer_data['LastPurchaseDate'].astype(str)
        
        customer_data.to_sql('Customers', conn, if_exists='append', index=False)
        print(f"✓ Inserted {len(customer_data)} customers")
        
    def populate_products(self, df, conn):
        """Populate Products table"""
        print("Populating Products table...")
        
        # Get unique products with most common price
        product_data = df.groupby('StockCode').agg({
            'Description': 'first',
            'UnitPrice': 'median'  # Use median price
        }).reset_index()
        
        product_data.to_sql('Products', conn, if_exists='append', index=False)
        print(f"✓ Inserted {len(product_data)} products")
        
    def populate_invoices(self, df, conn):
        """Populate Invoices table"""
        print("Populating Invoices table...")
        
        invoice_data = df.groupby('InvoiceNo').agg({
            'CustomerID': 'first',
            'InvoiceDate': 'first',
            'Country': 'first'
        }).reset_index()
        
        invoice_data['InvoiceDate'] = invoice_data['InvoiceDate'].astype(str)
        
        invoice_data.to_sql('Invoices', conn, if_exists='append', index=False)
        print(f"✓ Inserted {len(invoice_data)} invoices")
        
    def populate_invoice_items(self, df, conn):
        """Populate InvoiceItems table"""
        print("Populating InvoiceItems table...")
        
        items_data = df[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'TotalPrice']].copy()
        
        # Insert in batches to handle large datasets
        batch_size = 10000
        for i in range(0, len(items_data), batch_size):
            batch = items_data.iloc[i:i+batch_size]
            batch.to_sql('InvoiceItems', conn, if_exists='append', index=False)
            print(f"  Inserted batch {i//batch_size + 1} ({len(batch)} items)")
        
        print(f"✓ Inserted {len(items_data)} invoice items")
        
    def populate_database(self, csv_path='data/raw/online_retail.csv'):
        """Main function to populate all tables"""
        print("\n" + "="*60)
        print("POPULATING DATABASE FROM CSV")
        print("="*60 + "\n")
        
        # Load and clean data
        df = self.load_and_clean_data(csv_path)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Populate tables in order (respecting foreign keys)
            self.populate_customers(df, conn)
            self.populate_products(df, conn)
            self.populate_invoices(df, conn)
            self.populate_invoice_items(df, conn)
            
            conn.commit()
            print("\n✓ Database populated successfully!")
            
            # Print summary statistics
            self.print_database_stats(conn)
            
        except Exception as e:
            conn.rollback()
            print(f"✗ Error populating database: {e}")
            raise
        finally:
            conn.close()
            
    def print_database_stats(self, conn):
        """Print database statistics"""
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        
        cursor = conn.cursor()
        
        tables = ['Customers', 'Products', 'Invoices', 'InvoiceItems']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table:20s}: {count:,} records")
        
        print("="*60 + "\n")

def main():
    """Main execution function"""
    db = RetailDatabase()
    
    # Check if CSV exists
    csv_path = 'data/raw/online_retail.csv'
    if not os.path.exists(csv_path):
        print(f"✗ Error: CSV file not found at {csv_path}")
        print("\nPlease download the UCI Online Retail dataset from:")
        print("https://archive.ics.uci.edu/ml/datasets/online+retail")
        print("or Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset")
        print("\nPlace the file in: data/raw/online_retail.csv")
        return
    
    # Create database
    db.create_database()
    
    # Populate database
    db.populate_database(csv_path)
    
    print("\n✓ Database initialization complete!")
    print(f"Database location: {db.db_path}")

if __name__ == "__main__":
    main()