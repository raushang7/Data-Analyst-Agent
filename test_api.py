#!/usr/bin/env python3
"""
Test script for the Data Analyst Agent
"""

import requests
import json
import tempfile
import os

PORT = os.getenv("PORT", "3000")
BASE_URL = f"http://localhost:{PORT}"

def test_api():
    """Test the API with sample data"""
    
    # Use the local questions.txt file
    questions_file = "questions.txt"
    
    if not os.path.exists(questions_file):
        print(f"Error: {questions_file} not found in current directory")
        return
    
    try:
        # Test the API
        url = f"{BASE_URL}/api/"
        
        with open(questions_file, 'rb') as f:
            files = {'questions.txt': f}
            response = requests.post(url, files=files, timeout=180)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Got {len(result) if isinstance(result, list) else 'object'} result(s)")
            
            # Check if we got the expected format
            if isinstance(result, list) and len(result) >= 4:
                print(f"1. Movies before 2000: {result[0]}")
                print(f"2. Earliest film: {result[1]}")
                print(f"3. Correlation: {result[2]}")
                print(f"4. Visualization: {'Present' if result[3] and isinstance(result[3], str) and result[3].startswith('data:image') else 'Missing'}")
        else:
            print(f"Error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API. Make sure the server is running on {BASE_URL}")
    except Exception as e:
        print(f"Error: {e}")

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

if __name__ == "__main__":
    print("Testing Data Analyst Agent API...")
    print("\n1. Health Check:")
    test_health()
    
    print("\n2. API Test:")
    test_api()
