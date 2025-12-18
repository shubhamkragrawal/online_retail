"""
API Testing Script
Tests the FastAPI endpoints with sample data
"""

import requests
import json
from pprint import pprint

# API Configuration
API_URL = "http://localhost:8000"  # Change if deployed elsewhere

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print("Response:")
        pprint(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print("Response:")
        pprint(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction_low_risk():
    """Test prediction with low-risk customer"""
    print("\n" + "="*60)
    print("Testing Prediction - Low Risk Customer")
    print("="*60)
    
    # Low risk: recent purchase, high frequency, high spend
    customer_data = {
        "Recency": 10,
        "Frequency": 15,
        "Monetary": 2000.0,
        "InvoiceNo_nunique": 15,
        "Quantity_sum": 150.0,
        "Quantity_mean": 10.0,
        "TotalPrice_sum": 2000.0,
        "TotalPrice_mean": 133.33,
        "TotalPrice_std": 50.0,
        "StockCode_nunique": 25,
        "CustomerLifetime": 365,
        "AvgDaysBetweenPurchases": 24.0,
        "Country_United_Kingdom": 1,
        "Country_Germany": 0,
        "Country_France": 0,
        "Country_EIRE": 0,
        "Country_Spain": 0,
        "Country_Netherlands": 0,
        "Country_Belgium": 0,
        "Country_Switzerland": 0,
        "Country_Portugal": 0,
        "Country_Australia": 0
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            params={"customer_id": "CUST_LOW_RISK_001"}
        )
        print(f"Status Code: {response.status_code}")
        print("\nInput Customer Profile:")
        print(f"  Recency: {customer_data['Recency']} days")
        print(f"  Frequency: {customer_data['Frequency']} orders")
        print(f"  Monetary: £{customer_data['Monetary']}")
        print("\nPrediction:")
        pprint(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction_high_risk():
    """Test prediction with high-risk customer"""
    print("\n" + "="*60)
    print("Testing Prediction - High Risk Customer")
    print("="*60)
    
    # High risk: long time since purchase, low frequency, low spend
    customer_data = {
        "Recency": 180,
        "Frequency": 2,
        "Monetary": 100.0,
        "InvoiceNo_nunique": 2,
        "Quantity_sum": 10.0,
        "Quantity_mean": 5.0,
        "TotalPrice_sum": 100.0,
        "TotalPrice_mean": 50.0,
        "TotalPrice_std": 10.0,
        "StockCode_nunique": 3,
        "CustomerLifetime": 200,
        "AvgDaysBetweenPurchases": 100.0,
        "Country_United_Kingdom": 0,
        "Country_Germany": 1,
        "Country_France": 0,
        "Country_EIRE": 0,
        "Country_Spain": 0,
        "Country_Netherlands": 0,
        "Country_Belgium": 0,
        "Country_Switzerland": 0,
        "Country_Portugal": 0,
        "Country_Australia": 0
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            params={"customer_id": "CUST_HIGH_RISK_001"}
        )
        print(f"Status Code: {response.status_code}")
        print("\nInput Customer Profile:")
        print(f"  Recency: {customer_data['Recency']} days")
        print(f"  Frequency: {customer_data['Frequency']} orders")
        print(f"  Monetary: £{customer_data['Monetary']}")
        print("\nPrediction:")
        pprint(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    # Sample batch of 3 customers
    batch_data = {
        "customers": [
            {  # Low risk
                "Recency": 15,
                "Frequency": 10,
                "Monetary": 1500.0,
                "InvoiceNo_nunique": 10,
                "Quantity_sum": 100.0,
                "Quantity_mean": 10.0,
                "TotalPrice_sum": 1500.0,
                "TotalPrice_mean": 150.0,
                "TotalPrice_std": 30.0,
                "StockCode_nunique": 20,
                "CustomerLifetime": 300,
                "AvgDaysBetweenPurchases": 30.0,
                "Country_United_Kingdom": 1,
                "Country_Germany": 0,
                "Country_France": 0,
                "Country_EIRE": 0,
                "Country_Spain": 0,
                "Country_Netherlands": 0,
                "Country_Belgium": 0,
                "Country_Switzerland": 0,
                "Country_Portugal": 0,
                "Country_Australia": 0
            },
            {  # Medium risk
                "Recency": 60,
                "Frequency": 5,
                "Monetary": 500.0,
                "InvoiceNo_nunique": 5,
                "Quantity_sum": 40.0,
                "Quantity_mean": 8.0,
                "TotalPrice_sum": 500.0,
                "TotalPrice_mean": 100.0,
                "TotalPrice_std": 20.0,
                "StockCode_nunique": 10,
                "CustomerLifetime": 180,
                "AvgDaysBetweenPurchases": 36.0,
                "Country_United_Kingdom": 0,
                "Country_Germany": 1,
                "Country_France": 0,
                "Country_EIRE": 0,
                "Country_Spain": 0,
                "Country_Netherlands": 0,
                "Country_Belgium": 0,
                "Country_Switzerland": 0,
                "Country_Portugal": 0,
                "Country_Australia": 0
            },
            {  # High risk
                "Recency": 150,
                "Frequency": 2,
                "Monetary": 80.0,
                "InvoiceNo_nunique": 2,
                "Quantity_sum": 8.0,
                "Quantity_mean": 4.0,
                "TotalPrice_sum": 80.0,
                "TotalPrice_mean": 40.0,
                "TotalPrice_std": 5.0,
                "StockCode_nunique": 3,
                "CustomerLifetime": 160,
                "AvgDaysBetweenPurchases": 80.0,
                "Country_United_Kingdom": 0,
                "Country_Germany": 0,
                "Country_France": 1,
                "Country_EIRE": 0,
                "Country_Spain": 0,
                "Country_Netherlands": 0,
                "Country_Belgium": 0,
                "Country_Switzerland": 0,
                "Country_Portugal": 0,
                "Country_Australia": 0
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch_data
        )
        print(f"Status Code: {response.status_code}")
        print("\nBatch Prediction Results:")
        result = response.json()
        print(f"Total Customers: {result['total_customers']}")
        print(f"High Risk Count: {result['high_risk_count']}")
        print("\nIndividual Predictions:")
        for pred in result['predictions']:
            print(f"  {pred['customer_id']}: {pred['risk_level']} ({pred['churn_probability']:.2%})")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("STARTING API TESTS")
    print("="*60)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Prediction (Low Risk)", test_prediction_low_risk),
        ("Prediction (High Risk)", test_prediction_high_risk),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nTest '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_tests()
