#!/usr/bin/env python3
"""
Test script to demonstrate the hip keypoint correction functionality.
This script processes an input image, performs pose estimation and segmentation,
then applies the hip keypoint correction to improve accuracy.
"""

import argparse
import cv2
import numpy as np
import time
import os

from app.models.segmentation import SegmentationModel
from app.models.pose_estimation import PoseEstimationModel
from app.utils.hip_correction import HipKeypointCorrector
from app.utils.image_utils import load_image, resize_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Hip Keypoint Correction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='hip_correction_result.jpg', help='Path to output image')
    return parser.parse_args()

def main():
    """Main function to run the hip keypoint correction test."""
    # Parse arguments
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    
    # Load and resize image
    image = load_image(args.image)
    if image is None:
        print(f"Failed to load image from {args.image}")
        return
    
    image = resize_image(image, max_dimension=800)
    
    # Step 1: Perform segmentation to get the body mask
    print("Performing segmentation...")
    start_time = time.time()
    segmentation_model = SegmentationModel()
    segmented_image, mask = segmentation_model.segment_person(image)
    print(f"Segmentation completed in {time.time() - start_time:.2f} seconds")
    
    # Save segmentation result for visualization
    cv2.imwrite('debug_segmentation.jpg', segmented_image)
    
    # Step 2: Run pose estimation to get keypoints
    print("Detecting keypoints...")
    start_time = time.time()
    pose_model = PoseEstimationModel()
    keypoints = pose_model.detect_keypoints(segmented_image)
    print(f"Pose estimation completed in {time.time() - start_time:.2f} seconds")
    
    if keypoints is None:
        print("No person detected in the image or keypoints could not be extracted.")
        return
    
    # Save original keypoints visualization
    keypoint_vis = pose_model.visualize_keypoints(segmented_image, keypoints)
    cv2.imwrite('debug_keypoints.jpg', keypoint_vis)
    
    # Step 3: Apply hip keypoint correction
    print("Applying hip keypoint correction...")
    start_time = time.time()
    hip_corrector = HipKeypointCorrector(debug=True)
    corrected_keypoints = hip_corrector.correct_hip_keypoints(keypoints, mask, image)
    print(f"Hip keypoint correction completed in {time.time() - start_time:.2f} seconds")
    
    # Visualize corrected keypoints
    corrected_keypoint_vis = pose_model.visualize_keypoints(segmented_image, corrected_keypoints)
    cv2.imwrite('debug_corrected_keypoints.jpg', corrected_keypoint_vis)
    
    # Create a comparison visualization
    comparison = np.hstack((keypoint_vis, corrected_keypoint_vis))
    cv2.putText(comparison, "Original Keypoints", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "Corrected Keypoints", (keypoint_vis.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save the comparison image
    cv2.imwrite(args.output, comparison)
    print(f"Comparison image saved to {args.output}")
    
    # Calculate how much the keypoints were adjusted
    if ('left_hip' in keypoints and keypoints['left_hip'] and 
        'right_hip' in keypoints and keypoints['right_hip']):
        
        left_hip_original = keypoints['left_hip']
        right_hip_original = keypoints['right_hip']
        left_hip_corrected = corrected_keypoints['left_hip']
        right_hip_corrected = corrected_keypoints['right_hip']
        
        # Calculate hip widths
        original_width = abs(right_hip_original[0] - left_hip_original[0])
        corrected_width = abs(right_hip_corrected[0] - left_hip_corrected[0])
        
        # Calculate percentage change
        width_change_percent = ((corrected_width - original_width) / original_width) * 100
        
        print("\nHip Width Comparison:")
        print(f"  Original hip width: {original_width:.2f} pixels")
        print(f"  Corrected hip width: {corrected_width:.2f} pixels")
        print(f"  Width change: {width_change_percent:+.1f}%")
        
        # Suggest potential impact on body shape classification
        if width_change_percent > 15:
            print("\nPotential Impact on Classification:")
            print("  The significant increase in hip width may change classification from:")
            print("  - Rectangle/Inverted Triangle → Hourglass/Spoon")
            print("  - This correction is especially important for accurate body shape analysis")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 