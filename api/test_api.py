import requests
import os
import argparse
import time

def test_health_check(base_url):
    """Test the health check endpoint"""
    url = f"{base_url}/health"
    response = requests.get(url)
    print(f"Health Check Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)
    return response.status_code == 200

def test_prediction(base_url, image_path, height, is_female=True, debug=False):
    """Test the prediction endpoint"""
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found")
        return False
    
    url = f"{base_url}/predict"
    
    # Prepare form data
    data = {
        'height': str(height),
        'is_female': 'true' if is_female else 'false',
        'debug': 'true' if debug else 'false'
    }
    
    # Prepare file
    files = {
        'image': (os.path.basename(image_path), open(image_path, 'rb'), f'image/{os.path.splitext(image_path)[1][1:]}')
    }
    
    print(f"Sending request to {url}")
    print(f"Image: {image_path}")
    print(f"Height: {height} cm")
    print(f"Is Female: {is_female}")
    print(f"Debug Mode: {debug}")
    
    start_time = time.time()
    response = requests.post(url, data=data, files=files)
    elapsed_time = time.time() - start_time
    
    print(f"Response Status: {response.status_code}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Result:")
        print(f"Body Shape: {result.get('shape')}")
        print(f"Confidence: {result.get('confidence')}%")
        print(f"Description: {result.get('description')}")
        print(f"Fashion Pose Detected: {result.get('fashion_pose')}")
        
        # If debug mode is enabled, print features
        if debug and 'features' in result:
            print("\nBody Features:")
            for key, value in result['features'].items():
                print(f"{key}: {value}")
        
        # Print the URL for the result image
        image_url = f"{base_url}{result.get('result_image_url')}"
        print(f"\nResult Image URL: {image_url}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Body Shape API')
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='Base URL of the API')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--height', type=float, required=True, help='Person height in cm')
    parser.add_argument('--female', action='store_true', default=True, help='Specifies subject is female')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("=== Testing Body Shape API ===")
    
    # Test health check
    print("\nTesting Health Check:")
    if not test_health_check(args.url):
        print("Health check failed. Make sure the API is running.")
        return
    
    # Test prediction
    print("\nTesting Prediction:")
    test_prediction(
        base_url=args.url, 
        image_path=args.image, 
        height=args.height,
        is_female=args.female,
        debug=args.debug
    )

if __name__ == "__main__":
    main() 