import pandas as pd

df = pd.read_csv('data/raw/online_retail.csv', encoding='ISO-8859-1')
print(df.columns)
# data/raw/online_retail.csv
# cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'Price', 'CustomerID', 'Country']
# df.columns = cols
# df.rename(columns={'ï»¿InvoiceNo': 'InvoiceNo'}, inplace=True)
# df.to_csv('data/raw/online_retail.csv', index=False)
