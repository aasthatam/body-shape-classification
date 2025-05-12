#!/usr/bin/env python
"""
Sample image downloader for Body Shape Classification system.
Downloads sample images that are known to work with the system.
"""

import os
import sys
import argparse
import urllib.request
import ssl
from zipfile import ZipFile
from io import BytesIO

# Sample images URLs
SAMPLE_IMAGES_URL = "https://github.com/opencv/opencv/raw/master/samples/data/people"
# List of sample filenames to download
SAMPLE_FILES = [
    "lena.jpg",           # Standard test image
    "pedestrian.png",     # Full person
    "man_model.jpg",      # Full-body male model
    "woman_model.jpg",    # Full-body female model
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download sample images for testing')
    parser.add_argument('--output', type=str, default='static/sample_images', 
                      help='Output directory for sample images')
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
    return directory

def download_file(url, output_path):
    """Download a file from URL to the specified output path."""
    print(f"Downloading {url} to {output_path}...")
    
    try:
        # Create SSL context that doesn't verify certificates
        context = ssl._create_unverified_context()
        
        # Download the file
        with urllib.request.urlopen(url, context=context) as response:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
                
        print(f"✓ Successfully downloaded {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {str(e)}")
        return False

def download_opencv_sample(filename, output_dir):
    """Download a sample image from OpenCV samples."""
    url = f"{SAMPLE_IMAGES_URL}/{filename}"
    output_path = os.path.join(output_dir, filename)
    return download_file(url, output_path)

def download_default_sample(output_dir):
    """Download a default sample image that's guaranteed to work."""
    # This is a direct link to a sample image from a public dataset
    url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/examples/media/COCO_val2014_000000000241.jpg"
    output_path = os.path.join(output_dir, "sample_person.jpg")
    return download_file(url, output_path)

def main():
    """Main function to download sample images."""
    args = parse_args()
    
    print("Sample Image Downloader for Body Shape Classification")
    print("===================================================")
    
    # Ensure the output directory exists
    output_dir = ensure_directory(args.output)
    
    # Track download success
    success_count = 0
    
    # Try to download opencv samples
    print("\nDownloading sample images from OpenCV repository...")
    for filename in SAMPLE_FILES:
        if download_opencv_sample(filename, output_dir):
            success_count += 1
    
    # If no opencv samples could be downloaded, try the default sample
    if success_count == 0:
        print("\nFailed to download OpenCV samples. Trying alternative source...")
        if download_default_sample(output_dir):
            success_count += 1
    
    # Report results
    print("\nDownload Summary:")
    print(f"- {success_count} sample images downloaded to {output_dir}")
    
    if success_count > 0:
        print("\nYou can now test the system with one of these images:")
        print(f"python body_shape_prediction.py --image {output_dir}/sample_person.jpg --height 170")
        print("or")
        print(f"python test_predict.py --image {output_dir}/sample_person.jpg --height 170")
    else:
        print("\n✗ Failed to download any sample images.")
        print("Please download sample images manually and place them in the static/sample_images directory.")

if __name__ == "__main__":
    main() 