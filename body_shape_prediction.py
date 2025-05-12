import os
import argparse
import cv2
import time
import torch
import numpy as np

from app.models.segmentation import SegmentationModel
from app.models.pose_estimation import PoseEstimationModel
from app.models.body_shape_classifier import BodyShapeClassifier
from app.utils.feature_extraction import FeatureExtractor
from app.utils.hip_correction import HipKeypointCorrector
from app.utils.image_utils import load_image, resize_image, draw_results, is_fashion_pose



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Body Shape Classification')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--height', type=float, required=True, help='Person height in cm')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to output image')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--force-shape', type=str, help='Force a specific body shape classification')
    parser.add_argument('--female', action='store_true', default=True, help='Specifies subject is female (default: True)')
    parser.add_argument('--skip-correction', action='store_true', help='Skip silhouette-based hip correction')
    return parser.parse_args()

def predict_body_shape(image_path, height_cm, output_path=None, debug=False, force_shape=None, is_female=True, skip_correction=False):
    """
    Predict body shape from a single image and height.
    
    Args:
        image_path: Path to the input image
        height_cm: Height of the person in cm
        output_path: Path to save output image (optional)
        debug: Whether to save debug images and print detailed information
        force_shape: Override detection with a specific shape (optional)
        is_female: Whether the subject is female (affects body shape calculations)
        skip_correction: Skip silhouette-based hip keypoint correction
        
    Returns:
        Dictionary with shape prediction results
    """
    # Load and prepare image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return None
    
    # Resize image for processing
    image = resize_image(image, max_dimension=800)
    
    # Initialize models
    print("Initializing models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Background Removal (Instance Segmentation)
    print("Step 1: Performing background removal...")
    start_time = time.time()
    segmentation_model = SegmentationModel()
    segmented_image, mask = segmentation_model.segment_person(image)
    print(f"Segmentation completed in {time.time() - start_time:.2f} seconds")
    
    if debug:
        cv2.imwrite('debug_segmentation.jpg', segmented_image)
    
    # Step 2: Pose Estimation (Keypoint Detection)
    print("Step 2: Detecting body keypoints...")
    start_time = time.time()
    pose_model = PoseEstimationModel()
    keypoints = pose_model.detect_keypoints(segmented_image)
    print(f"Pose estimation completed in {time.time() - start_time:.2f} seconds")
    
    if keypoints is None:
        print("No person detected in the image or keypoints could not be extracted.")
        return None
    
    # Check if this is a fashion pose
    fashion_pose = is_fashion_pose(keypoints)
    
    if debug:
        keypoint_vis = pose_model.visualize_keypoints(segmented_image, keypoints)
        cv2.imwrite('debug_keypoints.jpg', keypoint_vis)
    
    # Step 2.5: Hip Keypoint Correction using Silhouette
    if not skip_correction:
        print("Step 2.5: Correcting hip keypoints using silhouette...")
        start_time = time.time()
        hip_corrector = HipKeypointCorrector(debug=debug)
        corrected_keypoints = hip_corrector.correct_hip_keypoints(keypoints, mask, image if debug else None)
        keypoints = corrected_keypoints  # Use the corrected keypoints for further processing
        print(f"Hip keypoint correction completed in {time.time() - start_time:.2f} seconds")
    
    # Step 3: Feature Extraction
    print("Step 3: Extracting geometric features...")
    start_time = time.time()
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(keypoints, height_cm)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    
    if features is None:
        print("Could not extract features from keypoints.")
        return None
    
    # Print extracted features
    if debug:
        print("\nExtracted Features:")
        for feature, value in features.items():
            print(f"{feature}: {value:.2f}" if isinstance(value, float) else f"{feature}: {value}")
    
    # Step 4: Body Shape Classification
    print("Step 4: Classifying body shape...")
    start_time = time.time()
    classifier = BodyShapeClassifier()
    
    # If a specific shape is forced, use that instead of prediction
    if force_shape:
        force_shape = force_shape.lower().capitalize()
        valid_shapes = ['Rectangle', 'Triangle', 'Inverted Triangle', 'Spoon', 'Hourglass']
        
        if force_shape in valid_shapes:
            print(f"Forcing body shape classification to: {force_shape}")
            description = classifier.get_shape_description(force_shape)
            shape_result = {
                'shape': force_shape,
                'confidence': 100.0,  # 100% confidence since it's manually specified
                'description': description
            }
        else:
            print(f"Warning: Invalid shape '{force_shape}'. Using automated prediction instead.")
            shape_result = classifier.classify(features)
    else:
        # Regular classification
        shape_result = classifier.classify(features)
        
        # Special handling for women in fashion photos
        if is_female and fashion_pose:
            waist_to_hip = features.get('waist_to_hip_ratio')
            shoulder_to_waist = features.get('shoulder_to_waist_ratio')
            
            # Women in fashion photography often display hourglass characteristics
            # even when measurements suggest otherwise due to pose/clothing
            if shape_result['shape'] in ['Inverted Triangle', 'Rectangle']:
                if waist_to_hip and waist_to_hip < 0.85:
                    # Defined waist relative to hips suggests hourglass
                    print("Female fashion pose with defined waist - reclassifying as Hourglass")
                    shape_result['shape'] = 'Hourglass'
                    shape_result['confidence'] = max(shape_result['confidence'], 80.0)
                    shape_result['description'] = classifier.get_shape_description('Hourglass')
                elif shoulder_to_waist and shoulder_to_waist > 1.15:
                    # Broader shoulders with defined waist still suggests hourglass
                    print("Female fashion pose with defined waist-shoulder ratio - reclassifying as Hourglass")
                    shape_result['shape'] = 'Hourglass'
                    shape_result['confidence'] = max(shape_result['confidence'], 75.0)
                    shape_result['description'] = classifier.get_shape_description('Hourglass')
    
    print(f"Classification completed in {time.time() - start_time:.2f} seconds")
    
    # Print results
    print("\n----- Body Shape Analysis Results -----")
    if shape_result['shape']:
        print(f"Predicted Body Shape: {shape_result['shape']}")
        print(f"Confidence: {shape_result['confidence']:.1f}%")
        if fashion_pose:
            print("Note: Fashion pose detected - measurements may be affected")
        if is_female:
            print("Note: Female-specific adjustments applied to hip measurements")
        if not skip_correction:
            print("Note: Hip measurements refined using silhouette data")
        print(f"\nDescription: {shape_result['description']}")
    else:
        print("Could not determine body shape from the image.")
    
    # Draw results on image
    result_image = draw_results(image, keypoints, shape_result, is_female=is_female)
    
    # Save output image if path is provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, result_image)
        print(f"\nResults saved to {output_path}")
    
    return {
        'shape': shape_result['shape'],
        'confidence': shape_result['confidence'],
        'description': shape_result['description'],
        'features': features,
        'keypoints': keypoints,
        'result_image': result_image,
        'fashion_pose': fashion_pose,
        'is_female': is_female
    }

def main():
    """Main function to run the body shape classification pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Run prediction
    predict_body_shape(
        image_path=args.image,
        height_cm=args.height,
        output_path=args.output,
        debug=args.debug,
        force_shape=args.force_shape,
        is_female=args.female,
        skip_correction=args.skip_correction
    )

if __name__ == "__main__":
    main() 