import cv2
import numpy as np
import os
import sys

def load_image(image_path):
    """
    Load an image from disk with enhanced error handling.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        OpenCV image or None if loading fails
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at path: {image_path}")
        print("Please check if the file path is correct.")
        return None
    
    # Check if it's a file
    if not os.path.isfile(image_path):
        print(f"Error: Path exists but is not a file: {image_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        print(f"Error: Image file is empty (0 bytes): {image_path}")
        return None
    
    # Check file extension
    _, ext = os.path.splitext(image_path)
    if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        print(f"Warning: Unusual file extension: {ext}")
        print("The file might not be a supported image format.")
    
    # Try loading with OpenCV
    try:
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            print("Possible causes:")
            print("1. The file format is not supported by OpenCV")
            print("2. The file is corrupted")
            print("3. The file contains no valid image data")
            print("\nTry running 'python image_test.py --image {image_path}' for more detailed diagnostics.")
            return None
            
        if image.size == 0:
            print(f"Error: Image loaded but contains no data: {image_path}")
            return None
            
        # Get basic image info
        height, width = image.shape[:2]
        print(f"Image loaded successfully: {width}x{height} pixels")
        
        return image
        
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        print(f"Try running 'python image_test.py --image {image_path}' for more detailed diagnostics.")
        return None

def resize_image(image, max_dimension=800):
    """
    Resize image while preserving aspect ratio.
    
    Args:
        image: OpenCV image
        max_dimension: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    if image is None:
        return None
        
    # Get image dimensions
    h, w = image.shape[:2]
    
    # If image is already smaller than max_dimension, return original
    if max(h, w) <= max_dimension:
        return image
    
    # Calculate new dimensions
    if h > w:
        new_h = max_dimension
        new_w = int(w * (new_h / h))
    else:
        new_w = max_dimension
        new_h = int(h * (new_w / w))
    
    # Resize image
    try:
        resized = cv2.resize(image, (new_w, new_h))
        return resized
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return image  # Return original if resizing fails

def is_fashion_pose(keypoints):
    """
    Detect if the image shows a typical fashion model pose that might affect measurements.
    
    Args:
        keypoints: Dictionary of detected keypoints
        
    Returns:
        Boolean indicating if a fashion pose is detected
    """
    # Check if we have the necessary keypoints
    if not all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        return False
    
    # Check for partial body (fashion images often show only upper body)
    has_lower_body = all(k in keypoints and keypoints[k] is not None 
                         for k in ['left_knee', 'right_knee', 'left_ankle', 'right_ankle'])
    
    if not has_lower_body:
        print("Fashion pose detected: Partial body shot (lower body not visible)")
        return True
    
    # Check for typical model poses:
    # 1. Hip twist (one hip higher than the other by a significant amount)
    if keypoints['left_hip'] and keypoints['right_hip']:
        hip_height_diff = abs(keypoints['left_hip'][1] - keypoints['right_hip'][1])
        shoulder_width = abs(keypoints['left_shoulder'][0] - keypoints['right_shoulder'][0])
        
        # If hip height difference is more than 15% of shoulder width, it's likely a posed stance
        if hip_height_diff > 0.15 * shoulder_width:
            print(f"Fashion pose detected: Hip twist/pose (height diff: {hip_height_diff}px)")
            return True
    
    return False

def draw_results(image, keypoints, shape_result, is_female=True):
    """
    Draw results on the image for visualization.
    
    Args:
        image: OpenCV image
        keypoints: Dictionary of body keypoints
        shape_result: Dictionary with shape classification results
        is_female: Whether the subject is female (for hip bias visualization)
        
    Returns:
        Image with visualization
    """
    if image is None:
        print("Error: Cannot draw results on None image")
        return None
        
    result_img = image.copy()
    
    # Check if this is a fashion pose
    fashion_pose = is_fashion_pose(keypoints)
    
    # Create a copy of keypoints that we can modify for visualization
    adjusted_keypoints = keypoints.copy()
    
    # Draw key measurement points
    key_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    
    # Check if we have original hip points (from silhouette correction)
    has_original_hips = 'original_left_hip' in keypoints and 'original_right_hip' in keypoints
    
    # Check if we have waist points (they should be added by feature extraction)
    has_waist_points = 'left_waist' in keypoints and 'right_waist' in keypoints
    
    # Draw all keypoints
    for point_name in list(adjusted_keypoints.keys()):
        if adjusted_keypoints[point_name] is not None:
            # Draw hip points (original and corrected)
            if 'hip' in point_name:
                if point_name == 'left_hip' or point_name == 'right_hip':
                    # Draw the corrected hip points in green
                    cv2.circle(result_img, adjusted_keypoints[point_name], 7, (0, 255, 0), -1)  # Green fill
                    cv2.circle(result_img, adjusted_keypoints[point_name], 9, (0, 255, 0), 2)  # Green outline
                    
                    # Draw labels for hip points
                    label_x = adjusted_keypoints[point_name][0]
                    label_y = adjusted_keypoints[point_name][1] - 15
                    cv2.putText(result_img, "Hip", (label_x - 10, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw original hip points if available
                elif has_original_hips and (point_name == 'original_left_hip' or point_name == 'original_right_hip'):
                    original_point = adjusted_keypoints[point_name]
                    cv2.circle(result_img, original_point, 5, (0, 0, 255), -1)  # Red fill
                    cv2.circle(result_img, original_point, 7, (0, 0, 255), 1)  # Red outline
                    
                    # Draw line to corrected hip point
                    if point_name == 'original_left_hip':
                        cv2.line(result_img, original_point, adjusted_keypoints['left_hip'], 
                                (0, 165, 255), 1, cv2.LINE_AA)  # Orange
                        
                        # Label as "Original"
                        label_x = original_point[0] - 40
                        label_y = original_point[1] - 15
                        cv2.putText(result_img, "Original", (label_x, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        cv2.line(result_img, original_point, adjusted_keypoints['right_hip'], 
                                (0, 165, 255), 1, cv2.LINE_AA)  # Orange
                        
                        # Label as "Original"
                        label_x = original_point[0] + 10
                        label_y = original_point[1] - 15
                        cv2.putText(result_img, "Original", (label_x, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw waist points in blue
            elif 'waist' in point_name and has_waist_points:
                cv2.circle(result_img, adjusted_keypoints[point_name], 7, (255, 0, 0), -1)  # Blue fill
                cv2.circle(result_img, adjusted_keypoints[point_name], 9, (255, 0, 0), 2)  # Blue outline
                
                # Label waist points
                label_x = adjusted_keypoints[point_name][0]
                label_y = adjusted_keypoints[point_name][1] - 15
                cv2.putText(result_img, "Waist", (label_x - 15, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw shoulder points in yellow
            elif 'shoulder' in point_name:
                cv2.circle(result_img, adjusted_keypoints[point_name], 7, (0, 255, 255), -1)  # Yellow fill
                cv2.circle(result_img, adjusted_keypoints[point_name], 9, (0, 255, 255), 2)  # Yellow outline
                
                # Label shoulder points
                label_x = adjusted_keypoints[point_name][0]
                label_y = adjusted_keypoints[point_name][1] - 15
                cv2.putText(result_img, "Shoulder", (label_x - 25, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw body shape outline
    if has_waist_points:
        # Draw straight vertical torso lines
        
        # 1. Shoulder to waist (left side)
        if 'left_shoulder' in adjusted_keypoints and 'left_waist' in adjusted_keypoints:
            cv2.line(result_img, 
                    adjusted_keypoints['left_shoulder'], 
                    adjusted_keypoints['left_waist'], 
                    (0, 255, 255), 2, cv2.LINE_AA)  # Yellow
        
        # 2. Shoulder to waist (right side)
        if 'right_shoulder' in adjusted_keypoints and 'right_waist' in adjusted_keypoints:
            cv2.line(result_img, 
                    adjusted_keypoints['right_shoulder'], 
                    adjusted_keypoints['right_waist'], 
                    (0, 255, 255), 2, cv2.LINE_AA)  # Yellow
        
        # 3. Waist to hip (left side)
        if 'left_waist' in adjusted_keypoints and 'left_hip' in adjusted_keypoints:
            cv2.line(result_img, 
                    adjusted_keypoints['left_waist'], 
                    adjusted_keypoints['left_hip'], 
                    (255, 0, 0), 2, cv2.LINE_AA)  # Blue
        
        # 4. Waist to hip (right side)
        if 'right_waist' in adjusted_keypoints and 'right_hip' in adjusted_keypoints:
            cv2.line(result_img, 
                    adjusted_keypoints['right_waist'], 
                    adjusted_keypoints['right_hip'], 
                    (255, 0, 0), 2, cv2.LINE_AA)  # Blue
    
    # Draw horizontal connecting lines
    
    # 1. Shoulder line (connecting left and right shoulder)
    if 'left_shoulder' in adjusted_keypoints and 'right_shoulder' in adjusted_keypoints:
        cv2.line(result_img, 
                adjusted_keypoints['left_shoulder'], 
                adjusted_keypoints['right_shoulder'], 
                (0, 255, 255), 2, cv2.LINE_AA)  # Yellow
    
    # 2. Waist line (connecting left and right waist)
    if has_waist_points:
        cv2.line(result_img, 
                adjusted_keypoints['left_waist'], 
                adjusted_keypoints['right_waist'], 
                (255, 0, 0), 2, cv2.LINE_AA)  # Blue
    
    # 3. Hip line (connecting left and right hip)
    if 'left_hip' in adjusted_keypoints and 'right_hip' in adjusted_keypoints:
        # Draw the original hip line in red if available
        if has_original_hips:
            cv2.line(result_img, 
                    keypoints['original_left_hip'], 
                    keypoints['original_right_hip'], 
                    (0, 0, 255), 1, cv2.LINE_AA)  # Red (thinner)
        
        # Draw the corrected hip line in green
        cv2.line(result_img, 
                adjusted_keypoints['left_hip'], 
                adjusted_keypoints['right_hip'], 
                (0, 255, 0), 2, cv2.LINE_AA)  # Green
    
    # Add shape classification result with more prominent styling
    if shape_result and shape_result['shape']:
        shape_text = f"Body Shape: {shape_result['shape']}"
        conf_text = f"Confidence: {shape_result['confidence']:.1f}%"
        
        # Draw a semi-transparent background for text
        text_bg = np.zeros_like(result_img)
        cv2.rectangle(text_bg, (0, 0), (400, 100), (255, 255, 255), -1)  # Extended for additional text
        
        # Blend the background with the original image
        alpha = 0.7
        cv2.addWeighted(result_img, alpha, text_bg, 1-alpha, 0, result_img)
        
        # Add text with thicker font
        cv2.putText(result_img, shape_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
        cv2.putText(result_img, conf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Add indicators for additional adjustments
        if is_female:
            cv2.putText(result_img, "Female Hip Adjustment Applied", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Add fashion pose warning if detected
        if fashion_pose:
            cv2.putText(result_img, "Fashion Pose Detected", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result_img 