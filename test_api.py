import requests
import json
import time
from typing import Dict, Any

def test_api(base_url: str = "http://localhost:8000"):
    """Test all API endpoints"""
    
    print("Testing API endpoints...\n")
    
    # Test health check
    print("1. Testing health check endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    
    # Test single prediction
    print("2. Testing single prediction endpoint...")
    transaction = {
        "amount": 1000.0,
        "time": 23.5,
        "n_transactions_day": 5,
        "merchant_category": "retail",
        "merchant_name": "Amazon",
        "customer_id": "CUST_12345",
        "transaction_id": "TXN_7891011"
    }
    
    response = requests.post(
        f"{base_url}/predict",
        json=transaction
    )
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    
    # Test batch prediction
    print("3. Testing batch prediction endpoint...")
    batch_transactions = {
        "transactions": [
            {
                "amount": 50.0,
                "time": 12.0,
                "n_transactions_day": 2,
                "merchant_category": "food",
                "merchant_name": "Starbucks",
                "customer_id": "CUST_12345",
                "transaction_id": "TXN_7891012"
            },
            {
                "amount": 5000.0,
                "time": 2.0,
                "n_transactions_day": 10,
                "merchant_category": "retail",
                "merchant_name": "Best Buy",
                "customer_id": "CUST_12345",
                "transaction_id": "TXN_7891013"
            }
        ]
    }
    
    response = requests.post(
        f"{base_url}/predict/batch",
        json=batch_transactions
    )
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Run tests
    test_api() 