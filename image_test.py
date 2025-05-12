#!/usr/bin/env python
"""
Image loading test utility.
This script helps diagnose issues with loading images for the body shape classification system.
"""

import os
import sys
import cv2
import argparse
from PIL import Image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test image loading')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to test')
    return parser.parse_args()

def test_file_existence(image_path):
    """Check if the file exists and is readable."""
    print(f"Testing file existence for: {image_path}")
    
    # Check if path exists
    if not os.path.exists(image_path):
        print("❌ Error: File does not exist")
        print(f"   The path '{image_path}' was not found on your system.")
        print("\nPossible solutions:")
        print("1. Check the file path for typos")
        print("2. Use an absolute path instead of a relative path")
        print("3. Make sure the image is in the correct directory")
        return False
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(image_path):
        print("❌ Error: Path exists but is not a file")
        print(f"   The path '{image_path}' is a directory, not a file.")
        return False
    
    # Check file size
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        print("❌ Error: File is empty (0 bytes)")
        return False
    
    print(f"✓ File exists and has size: {file_size} bytes")
    return True

def test_permissions(image_path):
    """Test if the file has read permissions."""
    print("\nTesting file permissions...")
    
    if not os.access(image_path, os.R_OK):
        print("❌ Error: No read permission")
        print(f"   The file '{image_path}' cannot be read due to permission restrictions.")
        print("\nPossible solutions:")
        print("1. Check file permissions (chmod +r filename on Linux/Mac)")
        print("2. Run the script with higher privileges")
        print("3. Move the file to a directory with proper permissions")
        return False
    
    print("✓ File has read permissions")
    return True

def test_opencv_loading(image_path):
    """Test loading the image with OpenCV."""
    print("\nTesting image loading with OpenCV (cv2)...")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Error: OpenCV could not load the image")
            print("   This usually means the file format is not supported or the file is corrupted.")
            print("\nPossible solutions:")
            print("1. Make sure the image is a valid format (JPG, PNG, BMP, etc.)")
            print("2. Try converting the image to JPG using another program")
            print("3. Check if the image can be opened in other applications")
            return False
        
        height, width, channels = image.shape
        print(f"✓ OpenCV loaded the image successfully")
        print(f"  - Image dimensions: {width}x{height} pixels")
        print(f"  - Channels: {channels}")
        return True
    except Exception as e:
        print(f"❌ Error: Exception when loading with OpenCV: {str(e)}")
        return False

def test_pillow_loading(image_path):
    """Test loading the image with Pillow (PIL)."""
    print("\nTesting image loading with Pillow (PIL)...")
    
    try:
        image = Image.open(image_path)
        width, height = image.size
        format_name = image.format
        mode = image.mode
        
        print(f"✓ Pillow loaded the image successfully")
        print(f"  - Image dimensions: {width}x{height} pixels")
        print(f"  - Format: {format_name}")
        print(f"  - Mode: {mode}")
        return True
    except Exception as e:
        print(f"❌ Error: Exception when loading with Pillow: {str(e)}")
        print("\nPossible solutions:")
        print("1. The image format might not be supported by Pillow")
        print("2. The image might be corrupted or incomplete")
        return False

def check_image_path_issues(image_path):
    """Check for common issues with image paths."""
    print("\nChecking for common path issues...")
    
    # Check for spaces in path
    if ' ' in image_path:
        print("⚠️ Warning: Path contains spaces")
        print("   Try enclosing the path in quotes when running the script.")
    
    # Check for correct slashes based on OS
    if sys.platform.startswith('win'):
        if '/' in image_path:
            print("⚠️ Warning: Path contains forward slashes on Windows")
            print("   Consider using backslashes (\\) or raw strings (r'path\\to\\file').")
    else:
        if '\\' in image_path:
            print("⚠️ Warning: Path contains backslashes on Unix/Mac")
            print("   Consider using forward slashes (/) instead.")
    
    # Check file extension
    _, ext = os.path.splitext(image_path)
    if not ext:
        print("⚠️ Warning: File has no extension")
        print("   Make sure the file has an appropriate image extension (.jpg, .png, etc.)")
    elif ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
        print(f"⚠️ Warning: Unusual file extension: {ext}")
        print("   The file extension suggests this might not be an image file.")
    
    # Check if path is absolute
    if not os.path.isabs(image_path):
        abs_path = os.path.abspath(image_path)
        print(f"ℹ️ Info: Using relative path. Absolute path is:")
        print(f"   {abs_path}")

def provide_recommendations():
    """Provide general recommendations for fixing image loading issues."""
    print("\n" + "="*50)
    print("RECOMMENDATIONS FOR FIXING IMAGE LOADING ISSUES")
    print("="*50)
    print("\n1. Supported formats: Use standard image formats like JPG or PNG")
    print("2. Image size: Make sure the image file is not too large (< 10MB)")
    print("3. Image dimensions: Images that are too large (e.g., > 4000x4000 pixels) may cause issues")
    print("4. Try converting: If problems persist, try converting the image to JPG with another program")
    print("5. Sample images: Use one of the sample images as a test to verify the system works")
    print("\nFor the body shape detection system:")
    print("- The image should show a full-body photo (head to toe)")
    print("- The person should be facing the camera")
    print("- The background should be relatively plain")
    print("- The person should be wearing fitted clothing for accurate measurements")

def main():
    """Main function to run the image loading test."""
    args = parse_args()
    image_path = args.image
    
    print("IMAGE LOADING TEST UTILITY")
    print("==========================")
    print(f"Testing image: {image_path}")
    
    # Check path issues
    check_image_path_issues(image_path)
    
    # Basic tests
    if not test_file_existence(image_path):
        provide_recommendations()
        return
    
    if not test_permissions(image_path):
        provide_recommendations()
        return
    
    # Loading tests
    opencv_success = test_opencv_loading(image_path)
    pillow_success = test_pillow_loading(image_path)
    
    if opencv_success and pillow_success:
        print("\n✅ SUCCESS: The image loaded successfully with both libraries.")
        print("   The image should work with the body shape classification system.")
    elif opencv_success:
        print("\n⚠️ PARTIAL SUCCESS: The image loaded with OpenCV but not with Pillow.")
        print("   This might still work with the body shape classification system, but could cause issues.")
    elif pillow_success:
        print("\n⚠️ PARTIAL SUCCESS: The image loaded with Pillow but not with OpenCV.")
        print("   This will likely cause problems with the body shape classification system.")
    else:
        print("\n❌ FAILURE: The image could not be loaded with either library.")
        print("   This image will not work with the body shape classification system.")
    
    provide_recommendations()

if __name__ == "__main__":
    main() 