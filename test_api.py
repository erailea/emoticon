#!/usr/bin/env python3
"""
Test script for the Emotion Analysis API
Tests all endpoints with the provided test.jpg image
"""

import requests
import json
import base64
import os
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"

def test_start_session() -> str:
    """Test the /start endpoint"""
    print("🚀 Testing /start endpoint...")
    
    response = requests.get(f"{BASE_URL}/start")
    
    if response.status_code == 200:
        data = response.json()
        session_id = data["sessionId"]
        print(f"✅ Session started successfully! SessionId: {session_id}")
        return session_id
    else:
        print(f"❌ Failed to start session: {response.status_code} - {response.text}")
        return None

def test_upload_inputs(session_id: str, image_path: str) -> bool:
    """Test the / (POST) endpoint"""
    print("\n📤 Testing / (POST) endpoint...")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare test data with multiple timestamps
    test_data = {
        "sessionId": session_id,
        "inputs": [
            {"timestamp": 1.5, "file": base64_image},
            {"timestamp": 3.2, "file": base64_image},
            {"timestamp": 5.8, "file": base64_image}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/", json=test_data)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Inputs uploaded successfully! Inputs count: {data['inputsCount']}")
        return True
    else:
        print(f"❌ Failed to upload inputs: {response.status_code} - {response.text}")
        return False

def test_end_session(session_id: str) -> Dict[str, Any]:
    """Test the /end endpoint"""
    print("\n🏁 Testing /end endpoint...")
    
    response = requests.get(f"{BASE_URL}/end", params={"sessionId": session_id})
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Session ended successfully!")
        print(f"📊 Results summary:")
        print(f"   SessionId: {data['sessionId']}")
        print(f"   Emotions detected: {list(data['aspects'].keys())}")
        
        # Show detailed results
        for emotion, values in data['aspects'].items():
            print(f"\n   {emotion.upper()}:")
            for result in values:
                print(f"     - Timestamp: {result['timestamp']}s, Value: {result['value']:.2f}%")
        
        return data
    else:
        print(f"❌ Failed to end session: {response.status_code} - {response.text}")
        return None

def test_health_check():
    """Test the /health endpoint"""
    print("\n🏥 Testing /health endpoint...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed! Status: {data['status']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code} - {response.text}")
        return False

def test_root_endpoint():
    """Test the / (GET) endpoint"""
    print("\n🏠 Testing / (GET) endpoint...")
    
    response = requests.get(f"{BASE_URL}/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Root endpoint working! API: {data['message']}")
        return True
    else:
        print(f"❌ Root endpoint failed: {response.status_code} - {response.text}")
        return False

def run_full_test():
    """Run complete API test suite"""
    print("🧪 Starting Emotion Analysis API Test Suite")
    print("=" * 50)
    
    # Test image path
    image_path = "test.jpg"
    
    # Test health check
    if not test_health_check():
        print("❌ Health check failed, stopping tests")
        return False
    
    # Test root endpoint
    if not test_root_endpoint():
        print("❌ Root endpoint failed, stopping tests")
        return False
    
    # Test start session
    session_id = test_start_session()
    if not session_id:
        print("❌ Failed to start session, stopping tests")
        return False
    
    # Test upload inputs
    if not test_upload_inputs(session_id, image_path):
        print("❌ Failed to upload inputs, stopping tests")
        return False
    
    # Test end session
    results = test_end_session(session_id)
    if not results:
        print("❌ Failed to end session, stopping tests")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
    print("📁 Check the 'storage' directory for persisted session data")
    
    return True

if __name__ == "__main__":
    try:
        run_full_test()
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API server is running on http://localhost:8000")
        print("💡 Start the server with: python main.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}") 