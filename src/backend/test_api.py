"""
Test script for API endpoints
"""
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint"""
    print("\n" + "="*50)
    print("Testing Root Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_health():
    """Test health check endpoints"""
    print("\n" + "="*50)
    print("Testing Health Check Endpoints")
    print("="*50)
    
    endpoints = [
        "/api/v1/health/",
        "/api/v1/health/services",
        "/api/v1/health/ready",
        "/api/v1/health/live"
    ]
    
    for endpoint in endpoints:
        response = requests.get(f"{BASE_URL}{endpoint}")
        print(f"\n{endpoint}")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2, default=str)}")
        else:
            print(f"  Error: {response.text}")


def test_sensor_data():
    """Test sensor data endpoints"""
    print("\n" + "="*50)
    print("Testing Sensor Data Endpoints")
    print("="*50)
    
    # Test realtime
    print("\nGET /api/v1/sensor-data/realtime")
    response = requests.get(f"{BASE_URL}/api/v1/sensor-data/realtime", params={"limit": 10})
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Count: {data.get('count', 0)}")
    else:
        print(f"  Error: {response.text}")
    
    # Test historical
    print("\nGET /api/v1/sensor-data/historical")
    start_time = (datetime.now() - timedelta(days=1)).isoformat()
    end_time = datetime.now().isoformat()
    
    response = requests.get(
        f"{BASE_URL}/api/v1/sensor-data/historical",
        params={
            "start_time": start_time,
            "end_time": end_time,
            "limit": 10
        }
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Count: {data.get('count', 0)}")
    else:
        print(f"  Error: {response.text}")
    
    # Test analytics
    print("\nGET /api/v1/sensor-data/analytics/RIG_01")
    response = requests.get(f"{BASE_URL}/api/v1/sensor-data/analytics/RIG_01")
    print(f"  Status: {response.status_code}")
    if response.status_code in [200, 404]:
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2, default=str)}")
        else:
            print(f"  Message: {response.json()}")


def test_predictions():
    """Test prediction endpoints"""
    print("\n" + "="*50)
    print("Testing Prediction Endpoints")
    print("="*50)
    
    # Test anomaly detection
    print("\nPOST /api/v1/predictions/anomaly-detection")
    sensor_data = {
        "rig_id": "RIG_01",
        "bit_temperature": 105.0,
        "motor_temperature": 95.0,
        "vibration_level": 2.5,
        "power_consumption": 280.0
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predictions/anomaly-detection",
        json=sensor_data
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Result: {json.dumps(data, indent=2)}")
    else:
        print(f"  Error: {response.text}")


def test_maintenance():
    """Test maintenance endpoints"""
    print("\n" + "="*50)
    print("Testing Maintenance Endpoints")
    print("="*50)
    
    # Test alerts
    print("\nGET /api/v1/maintenance/alerts")
    response = requests.get(
        f"{BASE_URL}/api/v1/maintenance/alerts",
        params={"hours": 24}
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Alerts count: {len(data)}")
    else:
        print(f"  Error: {response.text}")
    
    # Test schedule
    print("\nGET /api/v1/maintenance/schedule")
    response = requests.get(f"{BASE_URL}/api/v1/maintenance/schedule")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Schedule count: {len(data)}")
    else:
        print(f"  Error: {response.text}")


def test_config():
    """Test configuration endpoints"""
    print("\n" + "="*50)
    print("Testing Configuration Endpoints")
    print("="*50)
    
    # Test well profiles
    print("\nGET /api/v1/config/well-profiles")
    response = requests.get(f"{BASE_URL}/api/v1/config/well-profiles")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Profiles count: {len(data)}")
    else:
        print(f"  Error: {response.text}")
    
    # Test parameters
    print("\nGET /api/v1/config/parameters")
    response = requests.get(f"{BASE_URL}/api/v1/config/parameters")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Categories: {list(data.keys())}")
    else:
        print(f"  Error: {response.text}")


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("API Endpoint Testing")
    print("="*50)
    print(f"Testing API at: {BASE_URL}")
    print(f"Make sure the API server is running!")
    
    try:
        # Test root
        test_root()
        
        # Test health
        test_health()
        
        # Test sensor data
        test_sensor_data()
        
        # Test predictions
        test_predictions()
        
        # Test maintenance
        test_maintenance()
        
        # Test config
        test_config()
        
        print("\n" + "="*50)
        print("Testing Complete!")
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server!")
        print("Please make sure the API is running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

