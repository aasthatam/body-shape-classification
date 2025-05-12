import os
import sys
import uuid
import cv2
import json
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path so we can import from the main app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import directly from module to avoid circular imports
if 'body_shape_prediction' in sys.modules:
    del sys.modules['body_shape_prediction']

# Import the necessary components with absolute imports
try:
    from app.models.segmentation import SegmentationModel
    from app.models.pose_estimation import PoseEstimationModel
    from app.models.body_shape_classifier import BodyShapeClassifier
    from app.utils.feature_extraction import FeatureExtractor
    from app.utils.hip_correction import HipKeypointCorrector
    from app.utils.image_utils import load_image, resize_image, draw_results, is_fashion_pose
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Import necessary modules for predict_body_shape
import torch
import time

# Setup Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Import function definition instead of circular import
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Body Shape Prediction API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Body shape prediction endpoint"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Get height from form data
        height = request.form.get('height')
        if not height:
            return jsonify({'error': 'Height is required'}), 400
        
        height = float(height)
        if height <= 0:
            return jsonify({'error': 'Height must be a positive number'}), 400
        
        # Optional parameters
        is_female = request.form.get('is_female', 'true').lower() == 'true'
        debug = request.form.get('debug', 'false').lower() == 'true'
        force_shape = request.form.get('force_shape', None)
        skip_correction = request.form.get('skip_correction', 'false').lower() == 'true'
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        input_filename = secure_filename(f"{unique_id}_{file.filename}")
        output_filename = f"{unique_id}_result.jpg"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        # Save the uploaded file
        file.save(input_path)
        
        # Run prediction
        result = predict_body_shape(
            image_path=input_path,
            height_cm=height,
            output_path=output_path,
            debug=debug,
            force_shape=force_shape,
            is_female=is_female,
            skip_correction=skip_correction
        )
        
        if result is None:
            return jsonify({
                'error': 'No person detected in the image. Please ensure the image contains a clear view of person',
                 'code': 'NO_PERSON_DETECTED'
                 }), 400
        
        # Prepare the response
        response = {
            'shape': result['shape'],
            'confidence': result['confidence'],
            'description': result['description'],
            'result_image_url': f'/results/{output_filename}',
            'fashion_pose': result['fashion_pose'],
            'is_female': result['is_female']
        }
        
        # Add features if debug is enabled
        if debug and 'features' in result:
            # Convert NumPy arrays to lists for JSON serialization
            features = {}
            for key, value in result['features'].items():
                if isinstance(value, np.ndarray):
                    features[key] = value.tolist()
                else:
                    features[key] = value
            response['features'] = features
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>', methods=['GET'])
def get_result_image(filename):
    """Serve result images"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 