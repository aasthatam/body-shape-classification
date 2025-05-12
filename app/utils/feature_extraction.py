import numpy as np
import math

class FeatureExtractor:
    def __init__(self):
        """
        Utility for extracting geometric features from body keypoints.
        """
        # Standard body proportion ratios (approximate)
        self.standard_proportions = {
            'waist_to_hip_ideal': 0.7,  # Ideal waist-to-hip ratio
            'shoulder_to_hip_ideal': 1.0,  # Balanced shoulder-to-hip for hourglass
        }
        
        # Hip width adjustment factor for women (women typically have wider hips)
        self.hip_width_adjustment = 1.15  # Increase hip measurements by 15%
        
        # Waist vertical position ratio (from shoulders to hips)
        self.waist_vertical_ratio = 0.55  # Waist is typically ~40-45% down from shoulders to hips
    
    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: Tuple (x, y) for first point
            point2: Tuple (x, y) for second point
            
        Returns:
            Distance between points
        """
        if point1 is None or point2 is None:
            return None
            
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def calculate_waist_points(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """
        Calculate anatomically correct waist points using vertical lines down from shoulders.
        
        Args:
            left_shoulder: Tuple (x, y) for left shoulder point
            right_shoulder: Tuple (x, y) for right shoulder point
            left_hip: Tuple (x, y) for left hip point
            right_hip: Tuple (x, y) for right hip point
            
        Returns:
            Tuple of (left_waist, right_waist) points
        """
        if None in (left_shoulder, right_shoulder, left_hip, right_hip):
            return None, None
        
        # Calculate vertical distance between shoulders and hips
        left_vertical_dist = left_hip[1] - left_shoulder[1]
        right_vertical_dist = right_hip[1] - right_shoulder[1]
        
        # Calculate waist y-position (55% down from shoulders to hips)
        left_waist_y = int(left_shoulder[1] + left_vertical_dist * self.waist_vertical_ratio)
        right_waist_y = int(right_shoulder[1] + right_vertical_dist * self.waist_vertical_ratio)
        
        # Create waist points directly below shoulders (vertical alignment)
        left_waist = (left_shoulder[0], left_waist_y)
        right_waist = (right_shoulder[0], right_waist_y)
        
        return left_waist, right_waist
    
    def adjust_hip_width(self, hip_width, waist_width, image_width, keypoints):
        """
        Adjust hip width measurement for form-fitting clothing or cropped images.
        
        Args:
            hip_width: Original calculated hip width
            waist_width: Calculated waist width
            image_width: Width of the image
            keypoints: Detected keypoints dictionary
            
        Returns:
            Adjusted hip width
        """
        # Check if hip keypoints have been corrected using silhouette
        silhouette_corrected = ('original_left_hip' in keypoints and 'original_right_hip' in keypoints)
        
        if silhouette_corrected:
            # If the keypoints have been corrected using silhouette, we trust them more
            # and don't need to apply the standard adjustment factor
            print("Using silhouette-corrected hip measurements - skipping standard adjustment")
            return hip_width
        
        # Apply the women's hip width adjustment factor for uncorrected measurements
        hip_width = hip_width * self.hip_width_adjustment
        print(f"Applied women's hip width adjustment factor: {hip_width:.2f}")
        
        # Check if hip points seem compressed (common with form-fitting clothes)
        if hip_width < 1.05 * waist_width:
            # Form-fitting clothing likely compressing hip measurement
            
            # Look at overall silhouette if lower body is visible
            if 'left_knee' in keypoints and 'right_knee' in keypoints and keypoints['left_knee'] and keypoints['right_knee']:
                # If knees are visible, use them to estimate hip width
                knee_width = self.calculate_distance(keypoints['left_knee'], keypoints['right_knee'])
                if knee_width:
                    # Hip width is typically about 10-15% wider than knee width
                    # For women, we'll use a higher factor (25-30%)
                    estimated_hip_width = knee_width * 1.25
                    # Only use this if it's wider than the detected hip width
                    if estimated_hip_width > hip_width:
                        print(f"Adjusting hip width based on knee position: {hip_width:.2f} → {estimated_hip_width:.2f}")
                        return estimated_hip_width
            
            # If knees aren't visible or usable, use waist as reference
            # For women, hip measurement should be wider compared to waist
            adjusted_hip_width = waist_width * 1.3  # Increased from 1.15 to 1.3 for better accuracy
            
            print(f"Adjusting hip width for form-fitting clothing: {hip_width:.2f} → {adjusted_hip_width:.2f}")
            return adjusted_hip_width
                
        return hip_width
    
    def extract_features(self, keypoints, height_cm):
        """
        Extract geometric features from keypoints.
        
        Args:
            keypoints: Dictionary of keypoints with coordinates
            height_cm: Person's height in centimeters
            
        Returns:
            Dictionary of features and measurements
        """
        if not keypoints:
            return None
        
        # Check if we have all required keypoints
        required_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if not all(keypoints.get(k) for k in required_keypoints):
            return None
        
        # Get image dimensions (estimated from keypoint coordinates)
        all_x = [p[0] for p in keypoints.values() if p is not None and isinstance(p, tuple)]
        all_y = [p[1] for p in keypoints.values() if p is not None and isinstance(p, tuple)]
        image_width = max(all_x) - min(all_x) if all_x else 0
        image_height = max(all_y) - min(all_y) if all_y else 0
        
        print(f"Image dimensions (estimated): {image_width} x {image_height}")
        
        # Calculate pixel-based measurements
        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        left_hip = keypoints['left_hip']
        right_hip = keypoints['right_hip']
        
        # Check if silhouette correction was applied
        silhouette_corrected = ('original_left_hip' in keypoints and 'original_right_hip' in keypoints)
        if silhouette_corrected:
            original_left_hip = keypoints['original_left_hip']
            original_right_hip = keypoints['original_right_hip']
            original_hip_width = self.calculate_distance(original_left_hip, original_right_hip)
            print(f"Original hip width before silhouette correction: {original_hip_width:.2f} pixels")
        
        # Shoulder width (in pixels)
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
        
        # Hip width (in pixels)
        hip_width = self.calculate_distance(left_hip, right_hip)
        
        # Calculate anatomically correct waist points
        left_waist, right_waist = self.calculate_waist_points(
            left_shoulder, right_shoulder, left_hip, right_hip)
        
        # Store waist points in keypoints for visualization
        keypoints['left_waist'] = left_waist
        keypoints['right_waist'] = right_waist
        
        # Waist width (in pixels)
        waist_width = self.calculate_distance(left_waist, right_waist)
        
        # Adjust hip width for form-fitting clothing or partial visibility
        # Only if not already corrected by silhouette
        hip_width = self.adjust_hip_width(hip_width, waist_width, image_width, keypoints)
        
        # Print raw pixel measurements
        print(f"Raw Measurements (pixels):")
        print(f"  Shoulder width: {shoulder_width:.2f}")
        print(f"  Waist width: {waist_width:.2f}")
        print(f"  Hip width: {hip_width:.2f}")
        
        # Calculate pixel to cm ratio
        # Assume shoulder width is approximately 0.259 * height for average person
        estimated_shoulder_width_cm = 0.259 * height_cm
        pixel_to_cm = estimated_shoulder_width_cm / shoulder_width if shoulder_width else 1
        
        # Convert pixel measurements to cm
        shoulder_width_cm = shoulder_width * pixel_to_cm
        waist_width_cm = waist_width * pixel_to_cm
        hip_width_cm = hip_width * pixel_to_cm
        
        # Calculate key ratios
        shoulder_to_hip_ratio = shoulder_width / hip_width if hip_width else None
        waist_to_hip_ratio = waist_width / hip_width if hip_width and waist_width else None
        shoulder_to_waist_ratio = shoulder_width / waist_width if waist_width else None
        
        # Compile all features
        features = {
            'shoulder_width_cm': shoulder_width_cm,
            'waist_width_cm': waist_width_cm,
            'hip_width_cm': hip_width_cm,
            'shoulder_to_hip_ratio': shoulder_to_hip_ratio,
            'waist_to_hip_ratio': waist_to_hip_ratio,
            'shoulder_to_waist_ratio': shoulder_to_waist_ratio,
        }
        
        # Print calculated features
        print(f"Calculated Measurements:")
        print(f"  Shoulder width: {shoulder_width_cm:.2f} cm")
        print(f"  Waist width: {waist_width_cm:.2f} cm")
        print(f"  Hip width: {hip_width_cm:.2f} cm")
        
        return features 