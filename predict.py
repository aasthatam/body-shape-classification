import os
import argparse
import cv2
import time

from app.models.segmentation import SegmentationModel
from app.models.pose_estimation import PoseEstimationModel
from app.models.body_shape_classifier import BodyShapeClassifier
from app.utils.feature_extraction import FeatureExtractor
from app.utils.image_utils import load_image, resize_image, draw_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Body Shape Classification')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--height', type=float, required=True, help='Person height in cm')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to output image')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main function to run the body shape classification pipeline."""
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
        print("Failed to load image. Exiting.")
        return
    
    # Resize image for processing
    image = resize_image(image, max_dimension=800)
    
    # Initialize models
    print("Initializing models...")
    segmentation_model = SegmentationModel()
    pose_model = PoseEstimationModel()
    feature_extractor = FeatureExtractor()
    classifier = BodyShapeClassifier()
    
    # Step 1: Background Removal (Instance Segmentation)
    print("Step 1: Performing background removal...")
    start_time = time.time()
    segmented_image, mask = segmentation_model.segment_person(image)
    print(f"Segmentation completed in {time.time() - start_time:.2f} seconds")
    
    if args.debug:
        cv2.imwrite('debug_segmentation.jpg', segmented_image)
    
    # Step 2: Pose Estimation (Keypoint Detection)
    print("Step 2: Detecting body keypoints...")
    start_time = time.time()
    keypoints = pose_model.detect_keypoints(segmented_image)
    print(f"Pose estimation completed in {time.time() - start_time:.2f} seconds")
    
    if keypoints is None:
        print("No person detected in the image or keypoints could not be extracted.")
        return
    
    if args.debug:
        keypoint_vis = pose_model.visualize_keypoints(segmented_image, keypoints)
        cv2.imwrite('debug_keypoints.jpg', keypoint_vis)
    
    # Step 3: Feature Extraction
    print("Step 3: Extracting geometric features...")
    start_time = time.time()
    features = feature_extractor.extract_features(keypoints, args.height)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    
    if features is None:
        print("Could not extract features from keypoints.")
        return
    
    # Print extracted features
    if args.debug:
        print("\nExtracted Features:")
        for feature, value in features.items():
            print(f"{feature}: {value:.2f}" if isinstance(value, float) else f"{feature}: {value}")
    
    # Step 4: Body Shape Classification
    print("Step 4: Classifying body shape...")
    start_time = time.time()
    shape_result = classifier.classify(features)
    print(f"Classification completed in {time.time() - start_time:.2f} seconds")
    
    # Print results
    print("\n----- Body Shape Analysis Results -----")
    if shape_result['shape']:
        print(f"Predicted Body Shape: {shape_result['shape']}")
        print(f"Confidence: {shape_result['confidence']:.1f}%")
        print(f"\nDescription: {shape_result['description']}")
    else:
        print("Could not determine body shape from the image.")
    
    # Draw results on image
    result_image = draw_results(image, keypoints, shape_result)
    
    # Save output image
    cv2.imwrite(args.output, result_image)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 