import cv2
import numpy as np
import math

class HipKeypointCorrector:
    """
    Utility to correct hip keypoints using body silhouette from segmentation mask.
    """
    
    def __init__(self, debug=False):
        """
        Initialize the hip keypoint corrector.
        
        Args:
            debug: Whether to save debug images
        """
        self.debug = debug
    
    def find_silhouette_at_level(self, mask, y_level):
        """
        Find the leftmost and rightmost non-zero points at a given y-level in the mask.
        
        Args:
            mask: Binary segmentation mask (numpy array)
            y_level: The y-coordinate to search along
            
        Returns:
            Tuple of (left_x, right_x) or (None, None) if no points found
        """
        if y_level < 0 or y_level >= mask.shape[0]:
            return None, None
        
        # Get the row at y_level
        row = mask[y_level, :]
        
        # Find non-zero indices
        non_zero_indices = np.where(row > 0)[0]
        
        if len(non_zero_indices) == 0:
            return None, None
        
        # Get leftmost and rightmost points
        left_x = non_zero_indices[0]
        right_x = non_zero_indices[-1]
        
        return left_x, right_x
    
    def smooth_hip_level(self, mask, y_level, window_size=5):
        """
        Smooth the hip level by averaging multiple rows around the given y-level.
        
        Args:
            mask: Binary segmentation mask
            y_level: Center y-coordinate
            window_size: Number of rows to consider above and below y_level
            
        Returns:
            Tuple of (smoothed_left_x, smoothed_right_x)
        """
        half_window = window_size // 2
        start_y = max(0, y_level - half_window)
        end_y = min(mask.shape[0], y_level + half_window + 1)
        
        left_points = []
        right_points = []
        
        for y in range(start_y, end_y):
            left_x, right_x = self.find_silhouette_at_level(mask, y)
            if left_x is not None and right_x is not None:
                left_points.append(left_x)
                right_points.append(right_x)
        
        if not left_points or not right_points:
            return None, None
        
        # Return average values
        smoothed_left_x = int(sum(left_points) / len(left_points))
        smoothed_right_x = int(sum(right_points) / len(right_points))
        
        return smoothed_left_x, smoothed_right_x
    
    def correct_hip_keypoints(self, original_keypoints, mask, image=None):
        """
        Correct hip keypoints using the body silhouette from the segmentation mask.
        
        Args:
            original_keypoints: Dictionary of original keypoints from pose estimation
            mask: Binary segmentation mask (numpy array)
            image: Original image (for debug visualization)
            
        Returns:
            Dictionary with corrected keypoints
        """
        # Create a copy of the original keypoints
        corrected_keypoints = original_keypoints.copy()
        
        # Get the hip y-level (average of left and right hip y-coordinates)
        if ('left_hip' in original_keypoints and original_keypoints['left_hip'] is not None and
            'right_hip' in original_keypoints and original_keypoints['right_hip'] is not None):
            
            left_hip = original_keypoints['left_hip']
            right_hip = original_keypoints['right_hip']
            
            hip_y_level = (left_hip[1] + right_hip[1]) // 2
            
            # Find the body silhouette at the hip level
            silhouette_left_x, silhouette_right_x = self.smooth_hip_level(mask, hip_y_level)
            
            if silhouette_left_x is not None and silhouette_right_x is not None:
                # Calculate the center of the original hip points and the silhouette
                original_center_x = (left_hip[0] + right_hip[0]) // 2
                silhouette_center_x = (silhouette_left_x + silhouette_right_x) // 2
                
                # Calculate the x-offset to align the centers
                x_offset = silhouette_center_x - original_center_x
                
                # Create corrected hip points based on silhouette
                corrected_left_hip = (silhouette_left_x, hip_y_level)
                corrected_right_hip = (silhouette_right_x, hip_y_level)
                
                # Update the keypoints dictionary
                corrected_keypoints['left_hip'] = corrected_left_hip
                corrected_keypoints['right_hip'] = corrected_right_hip
                
                # Store original keypoints for reference
                corrected_keypoints['original_left_hip'] = left_hip
                corrected_keypoints['original_right_hip'] = right_hip
                
                # Calculate how much the keypoints were adjusted
                original_width = abs(right_hip[0] - left_hip[0])
                corrected_width = abs(corrected_right_hip[0] - corrected_left_hip[0])
                width_change_percent = ((corrected_width - original_width) / original_width) * 100
                
                print(f"Hip keypoints corrected using silhouette:")
                print(f"  Original hip width: {original_width} pixels")
                print(f"  Corrected hip width: {corrected_width} pixels")
                print(f"  Width change: {width_change_percent:.1f}%")
                
                # Generate debug visualization if requested
                if self.debug and image is not None:
                    debug_img = image.copy()
                    
                    # Draw original hip points in red
                    cv2.circle(debug_img, left_hip, 5, (0, 0, 255), -1)  # Red
                    cv2.circle(debug_img, right_hip, 5, (0, 0, 255), -1)  # Red
                    cv2.line(debug_img, left_hip, right_hip, (0, 0, 255), 2)  # Red
                    
                    # Draw corrected hip points in green
                    cv2.circle(debug_img, corrected_left_hip, 7, (0, 255, 0), -1)  # Green
                    cv2.circle(debug_img, corrected_right_hip, 7, (0, 255, 0), -1)  # Green
                    cv2.line(debug_img, corrected_left_hip, corrected_right_hip, (0, 255, 0), 2)  # Green
                    
                    # Draw the hip level line
                    cv2.line(debug_img, (0, hip_y_level), (debug_img.shape[1], hip_y_level), (255, 255, 0), 1)  # Yellow
                    
                    # Add text explaining the correction
                    text = f"Hip width: {original_width} -> {corrected_width} ({width_change_percent:+.1f}%)"
                    cv2.putText(debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imwrite('debug_hip_correction.jpg', debug_img)
        
        return corrected_keypoints 