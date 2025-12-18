#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Header
echo "================================================================"
echo "  Retail Customer Churn Classification - Setup Script"
echo "================================================================"
echo ""

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Create directory structure
print_status "Creating directory structure..."
mkdir -p data/raw data/processed
mkdir -p database
mkdir -p models
mkdir -p results
mkdir -p notebooks
mkdir -p experiments
mkdir -p api
mkdir -p streamlit
mkdir -p docs
print_success "Directory structure created"

# Create .gitkeep files to preserve empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch results/.gitkeep

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "Pip upgraded"

# Install dependencies
print_status "Installing dependencies (this may take a few minutes)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Check if dataset exists
print_status "Checking for dataset..."
if [ -f "data/raw/online_retail.csv" ]; then
    print_success "Dataset found: data/raw/online_retail.csv"
    
    # Initialize database
    print_status "Initializing database..."
    python database/init_db.py
    if [ $? -eq 0 ]; then
        print_success "Database initialized successfully"
    else
        print_error "Database initialization failed"
        exit 1
    fi
    
    # Generate ML dataset
    print_status "Generating ML dataset..."
    python data/feature_engineering.py
    if [ $? -eq 0 ]; then
        print_success "ML dataset generated successfully"
    else
        print_error "Feature engineering failed"
        exit 1
    fi
    
else
    print_warning "Dataset not found at data/raw/online_retail.csv"
    echo ""
    echo "Please download the UCI Online Retail Dataset from:"
    echo "  - UCI: https://archive.ics.uci.edu/ml/datasets/online+retail"
    echo "  - Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset"
    echo ""
    echo "Place the file at: data/raw/online_retail.csv"
    echo ""
    echo "After downloading, run this script again or run:"
    echo "  python database/init_db.py"
    echo "  python data/feature_engineering.py"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created"
        print_warning "Please edit .env file with your DagsHub credentials"
    else
        print_error ".env.example not found"
    fi
else
    print_warning ".env file already exists"
fi

# Summary
echo ""
echo "================================================================"
echo "  Setup Complete!"
echo "================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Ensure dataset is at: data/raw/online_retail.csv"
echo "   (Download if not present)"
echo ""
echo "2. Edit .env file with your DagsHub credentials:"
echo "   vi .env"
echo ""
echo "3. Run experiments:"
echo "   python experiments/run_experiments.py"
echo ""
echo "4. Track experiments with MLflow:"
echo "   python experiments/mlflow_tracking.py"
echo ""
echo "5. Test API locally:"
echo "   cd api && uvicorn main:app --reload"
echo ""
echo "6. Test Streamlit UI locally:"
echo "   cd streamlit && streamlit run app.py"
echo ""
echo "7. Deploy with Docker:"
echo "   docker-compose up --build"
echo ""
echo "================================================================"
echo ""

print_success "Setup completed successfully!"
echo ""
echo "To activate the virtual environment later, run:"
echo "  source venv/bin/activate"
echo ""