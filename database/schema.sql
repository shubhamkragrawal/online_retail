-- 3NF Normalized Database Schema for Online Retail Dataset
-- Supports customer churn classification problem

-- Table 1: Customers (Customer information)
CREATE TABLE IF NOT EXISTS Customers (
    CustomerID INTEGER PRIMARY KEY,
    Country TEXT NOT NULL,
    FirstPurchaseDate TEXT,
    LastPurchaseDate TEXT,
    TotalPurchases INTEGER DEFAULT 0,
    TotalSpent REAL DEFAULT 0.0
);

-- Table 2: Products (Product catalog)
CREATE TABLE IF NOT EXISTS Products (
    StockCode TEXT PRIMARY KEY,
    Description TEXT,
    UnitPrice REAL NOT NULL
);

-- Table 3: Invoices (Order/Invoice header)
CREATE TABLE IF NOT EXISTS Invoices (
    InvoiceNo TEXT PRIMARY KEY,
    CustomerID INTEGER NOT NULL,
    InvoiceDate TEXT NOT NULL,
    Country TEXT NOT NULL,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- Table 4: InvoiceItems (Line items for each invoice)
CREATE TABLE IF NOT EXISTS InvoiceItems (
    ItemID INTEGER PRIMARY KEY AUTOINCREMENT,
    InvoiceNo TEXT NOT NULL,
    StockCode TEXT NOT NULL,
    Quantity INTEGER NOT NULL,
    UnitPrice REAL NOT NULL,
    TotalPrice REAL NOT NULL,
    FOREIGN KEY (InvoiceNo) REFERENCES Invoices(InvoiceNo),
    FOREIGN KEY (StockCode) REFERENCES Products(StockCode)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_invoices_customer ON Invoices(CustomerID);
CREATE INDEX IF NOT EXISTS idx_invoices_date ON Invoices(InvoiceDate);
CREATE INDEX IF NOT EXISTS idx_items_invoice ON InvoiceItems(InvoiceNo);
CREATE INDEX IF NOT EXISTS idx_items_stock ON InvoiceItems(StockCode);

-- View for denormalized data (useful for ML feature extraction)
CREATE VIEW IF NOT EXISTS TransactionView AS
SELECT 
    ii.ItemID,
    i.InvoiceNo,
    i.CustomerID,
    i.InvoiceDate,
    i.Country,
    ii.StockCode,
    p.Description,
    ii.Quantity,
    ii.UnitPrice,
    ii.TotalPrice
FROM InvoiceItems ii
JOIN Invoices i ON ii.InvoiceNo = i.InvoiceNo
JOIN Products p ON ii.StockCode = p.StockCode
JOIN Customers c ON i.CustomerID = c.CustomerID;