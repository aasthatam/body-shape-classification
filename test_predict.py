import os
import argparse
import cv2
import numpy as np
import random
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Body Shape Classification')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--height', type=float, required=True, help='Person height in cm')
    parser.add_argument('--output', type=str, default='test_output.jpg', help='Path to output image')
    return parser.parse_args()

def load_image(image_path):
    """Load an image from file."""
    return cv2.imread(image_path)

def simulate_segmentation(image):
    """Simulate background removal."""
    print("Simulating background removal...")
    time.sleep(1)  # Simulate processing time
    
    # Simple simulation - darken background areas (this is just a visual example)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create an elliptical mask as a simple person silhouette
    center = (w//2, h//2)
    axes = (w//4, h//2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    # Apply mask
    mask_3ch = cv2.merge([mask, mask, mask])
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented

def simulate_keypoints(image):
    """Simulate keypoint detection."""
    print("Simulating keypoint detection...")
    time.sleep(1)  # Simulate processing time
    
    h, w = image.shape[:2]
    
    # Simulate keypoints - just for visualization
    keypoints = {
        'left_shoulder': (int(w*0.35), int(h*0.25)),
        'right_shoulder': (int(w*0.65), int(h*0.25)),
        'left_hip': (int(w*0.4), int(h*0.5)),
        'right_hip': (int(w*0.6), int(h*0.5)),
        'left_knee': (int(w*0.45), int(h*0.7)),
        'right_knee': (int(w*0.55), int(h*0.7)),
        'left_ankle': (int(w*0.45), int(h*0.9)),
        'right_ankle': (int(w*0.55), int(h*0.9))
    }
    
    return keypoints

def simulate_feature_extraction(keypoints, height_cm):
    """Simulate feature extraction."""
    print("Simulating feature extraction...")
    time.sleep(1)  # Simulate processing time
    
    # Calculate simulated measurements
    shoulder_width = abs(keypoints['right_shoulder'][0] - keypoints['left_shoulder'][0])
    hip_width = abs(keypoints['right_hip'][0] - keypoints['left_hip'][0])
    
    # Estimate waist position
    left_waist = ((keypoints['left_shoulder'][0] + keypoints['left_hip'][0]) // 2, 
                  (keypoints['left_shoulder'][1] + keypoints['left_hip'][1]) // 2)
    right_waist = ((keypoints['right_shoulder'][0] + keypoints['right_hip'][0]) // 2, 
                  (keypoints['right_shoulder'][1] + keypoints['right_hip'][1]) // 2)
    waist_width = abs(right_waist[0] - left_waist[0])
    
    # Scale to cm (simple simulation)
    pixel_to_cm = height_cm / (keypoints['right_ankle'][1] - keypoints['right_shoulder'][1])
    shoulder_width_cm = shoulder_width * pixel_to_cm
    waist_width_cm = waist_width * pixel_to_cm
    hip_width_cm = hip_width * pixel_to_cm
    
    # Calculate ratios
    shoulder_to_hip_ratio = shoulder_width / hip_width
    waist_to_hip_ratio = waist_width / hip_width
    shoulder_to_waist_ratio = shoulder_width / waist_width
    
    # Return features
    features = {
        'shoulder_width_cm': shoulder_width_cm,
        'waist_width_cm': waist_width_cm,
        'hip_width_cm': hip_width_cm,
        'shoulder_to_hip_ratio': shoulder_to_hip_ratio,
        'waist_to_hip_ratio': waist_to_hip_ratio,
        'shoulder_to_waist_ratio': shoulder_to_waist_ratio,
    }
    
    return features

def simulate_classification(features):
    """Simulate body shape classification."""
    print("Simulating body shape classification...")
    time.sleep(1)  # Simulate processing time
    
    # Extract ratios from features
    shoulder_to_hip = features['shoulder_to_hip_ratio']
    waist_to_hip = features['waist_to_hip_ratio']
    
    # Simple classification logic
    if 0.9 <= shoulder_to_hip <= 1.1 and waist_to_hip < 0.8:
        shape = 'Hourglass'
    elif 0.9 <= shoulder_to_hip <= 1.1 and waist_to_hip >= 0.8:
        shape = 'Rectangle'
    elif shoulder_to_hip < 0.9:
        if waist_to_hip < 0.8:
            shape = 'Spoon'
        else:
            shape = 'Triangle'
    elif shoulder_to_hip > 1.1:
        shape = 'Inverted Triangle'
    else:
        shape = 'Rectangle'  # Default
    
    # Descriptions for each shape
    descriptions = {
        'Rectangle': (
            "Your shoulders, waist, and hips are about the same width, "
            "creating a straight up-and-down appearance. "
            "Styling tip: Create curves with peplum tops, belted waists, "
            "and full or A-line skirts."
        ),
        'Triangle': (
            "Your hips are wider than your shoulders, tapering upward "
            "to a narrower upper body. "
            "Styling tip: Balance your silhouette with statement shoulders, "
            "structured jackets, and A-line skirts."
        ),
        'Inverted Triangle': (
            "Your shoulders are wider than your hips, creating a "
            "broad upper body that tapers down. "
            "Styling tip: Balance your proportions with fuller skirts, "
            "wide-leg pants, and details at the hip area."
        ),
        'Spoon': (
            "Similar to a triangle, but with a more defined waist and "
            "a greater difference between waist and hip measurements. "
            "Styling tip: Highlight your waist and choose tops that balance "
            "your hip area."
        ),
        'Hourglass': (
            "Your shoulders and hips are about the same width with a "
            "significantly smaller waist, creating curves. "
            "Styling tip: Highlight your waist with fitted or belted pieces, "
            "and choose clothes that follow your curves."
        )
    }
    
    # Generate a confidence value
    confidence = random.uniform(75, 98)
    
    return {
        'shape': shape,
        'confidence': confidence,
        'description': descriptions[shape]
    }

def draw_results(image, keypoints, shape_result):
    """Draw keypoints and classification results on the image."""
    result_img = image.copy()
    
    # Draw keypoints
    for name, point in keypoints.items():
        cv2.circle(result_img, point, 5, (0, 255, 0), -1)
    
    # Draw connections
    connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip')
    ]
    
    for start_name, end_name in connections:
        cv2.line(result_img, keypoints[start_name], keypoints[end_name], (0, 0, 255), 2)
    
    # Draw waist points and connection
    left_waist = ((keypoints['left_shoulder'][0] + keypoints['left_hip'][0]) // 2, 
                  (keypoints['left_shoulder'][1] + keypoints['left_hip'][1]) // 2)
    right_waist = ((keypoints['right_shoulder'][0] + keypoints['right_hip'][0]) // 2, 
                   (keypoints['right_shoulder'][1] + keypoints['right_hip'][1]) // 2)
    
    cv2.circle(result_img, left_waist, 5, (255, 0, 0), -1)
    cv2.circle(result_img, right_waist, 5, (255, 0, 0), -1)
    cv2.line(result_img, left_waist, right_waist, (255, 0, 0), 2)
    
    # Add shape classification result
    shape_text = f"Body Shape: {shape_result['shape']}"
    conf_text = f"Confidence: {shape_result['confidence']:.1f}%"
    
    cv2.putText(result_img, shape_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_img, conf_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result_img

def main():
    """Main function for test script."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load image
    print(f"Loading image from {args.image}")
    image = load_image(args.image)
    if image is None:
        print(f"Failed to load image from {args.image}")
        return
    
    print("\nRunning Body Shape Classification Test (Simulation)")
    print("------------------------------------------------")
    
    # Step 1: Simulated Background Removal
    segmented_image = simulate_segmentation(image)
    
    # Step 2: Simulated Pose Estimation
    keypoints = simulate_keypoints(segmented_image)
    
    # Step 3: Simulated Feature Extraction
    features = simulate_feature_extraction(keypoints, args.height)
    
    # Print features
    print("\nExtracted Features:")
    for feature, value in features.items():
        print(f"{feature}: {value:.2f}")
    
    # Step 4: Simulated Body Shape Classification
    shape_result = simulate_classification(features)
    
    # Print results
    print("\n----- Body Shape Analysis Results -----")
    print(f"Predicted Body Shape: {shape_result['shape']}")
    print(f"Confidence: {shape_result['confidence']:.1f}%")
    print(f"\nDescription: {shape_result['description']}")
    
    # Draw results on image
    result_image = draw_results(image, keypoints, shape_result)
    
    # Save output image
    cv2.imwrite(args.output, result_image)
    print(f"\nTest results saved to {args.output}")
    print("\nNote: This is a simulation for testing. For actual predictions, use predict.py")

if __name__ == "__main__":
    main() 