import os
import argparse
import cv2
import numpy as np
from app.utils.image_utils import resize_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image Preparation for Body Shape Classification')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output image (default: prepared_[input filename])')
    parser.add_argument('--resize', type=int, default=800, help='Maximum dimension size for resizing')
    return parser.parse_args()

def prepare_image(image_path, output_path=None, max_dimension=800):
    """
    Prepare an image for body shape classification by:
    1. Resizing to a manageable dimension
    2. Ensuring the person is clearly visible
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the prepared image
        max_dimension: Maximum dimension to resize the image to
        
    Returns:
        Path to the prepared image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Resize image
    resized = resize_image(image, max_dimension=max_dimension)
    
    # If no output path is specified, create one based on the input path
    if output_path is None:
        dirname = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        filename, ext = os.path.splitext(basename)
        output_path = os.path.join(dirname, f"prepared_{filename}{ext}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the prepared image
    cv2.imwrite(output_path, resized)
    print(f"Prepared image saved to: {output_path}")
    
    # Print guidelines
    print("\nImage Preparation Guidelines:")
    print("1. Ensure the person is standing upright, facing the camera")
    print("2. The entire body (head to toe) should be visible")
    print("3. Wear form-fitting clothing for more accurate measurements")
    print("4. Stand against a plain background if possible")
    print("5. Make sure the image is well-lit")
    
    return output_path

def main():
    # Parse arguments
    args = parse_args()
    
    # Prepare the image
    prepare_image(args.image, args.output, args.resize)

if __name__ == "__main__":
    main() 