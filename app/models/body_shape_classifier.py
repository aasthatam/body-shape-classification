import numpy as np

class BodyShapeClassifier:
    def __init__(self):
        """
        Classifier for determining body shape from geometric features.
        
        Body Shape Categories:
        - Rectangle: Shoulders, waist and hips are similar width (balanced)
        - Triangle (Pear): Hips are wider than shoulders
        - Inverted Triangle: Shoulders are wider than hips
        - Spoon: Similar to triangle but with a more defined waist
        - Hourglass: Shoulders and hips are similar with a clearly defined waist
        """
        # Define body shape categories
        self.body_shapes = [
            'Rectangle', 
            'Triangle', 
            'Inverted Triangle', 
            'Spoon', 
            'Hourglass'
        ]
        
    def classify(self, features):
        """
        Classify body shape based on extracted features.
        
        Args:
            features: Dictionary of body measurements and ratios
            
        Returns:
            Dictionary with predicted body shape and confidence
        """
        if not features:
            return {
                'shape': None,
                'confidence': 0,
                'description': None
            }
        
        # Extract key ratios for classification
        shoulder_to_hip = features.get('shoulder_to_hip_ratio')
        waist_to_hip = features.get('waist_to_hip_ratio')
        shoulder_to_waist = features.get('shoulder_to_waist_ratio')
        
        # If any key ratio is missing, return None
        if not all([shoulder_to_hip, waist_to_hip, shoulder_to_waist]):
            return {
                'shape': None,
                'confidence': 0,
                'description': None
            }
        
        # Print debug information about the features
        print(f"Debug - Feature Analysis:")
        print(f"  Shoulder-to-Hip Ratio: {shoulder_to_hip:.2f}")
        print(f"  Waist-to-Hip Ratio: {waist_to_hip:.2f}")
        print(f"  Shoulder-to-Waist Ratio: {shoulder_to_waist:.2f}")
        
        # Classification logic and confidence calculation
        shape = None
        confidence = 0
        
        # Hourglass: shoulders and hips are aligned, with significantly smaller waist
        # Further relaxed threshold for shoulder-to-hip ratio for women who typically have wider hips
        # Women with hourglass figures often have shoulder-to-hip ratio below 1.0
        if (0.8 <= shoulder_to_hip <= 1.15) and (waist_to_hip < 0.85):
            # Calculate how defined the waist is compared to hips and shoulders
            waist_definition = 2.0 - (waist_to_hip + waist_to_hip/shoulder_to_waist)
            hourglass_confidence = waist_definition * (1.0 - abs(1.0 - shoulder_to_hip))
            if hourglass_confidence > confidence:
                shape = 'Hourglass'
                confidence = hourglass_confidence
        
        # Enhanced hourglass detection for fitted clothing that may distort measurements
        # More aggressive waist definition check for women
        if (shoulder_to_hip > 0.9) and (waist_to_hip < 0.8) and (shoulder_to_waist > 1.15):
            enhanced_hourglass_confidence = 0.9 * (shoulder_to_waist - 1.0) * (1.0 - waist_to_hip)
            if enhanced_hourglass_confidence > confidence:
                shape = 'Hourglass'
                confidence = enhanced_hourglass_confidence
        
        # Rectangle: shoulders, waist and hips are similar widths
        # Adjusted for women who may have slightly wider hips even in rectangle shapes
        if (0.85 <= shoulder_to_hip <= 1.1) and (0.8 <= waist_to_hip):
            rectangle_confidence = 1.0 - abs(1.0 - shoulder_to_hip) - abs(0.9 - waist_to_hip)
            if rectangle_confidence > confidence:
                shape = 'Rectangle'
                confidence = rectangle_confidence
        
        # Triangle (Pear): hips are wider than shoulders
        # More common in women, so we use a more generous threshold
        if (shoulder_to_hip < 0.9):
            triangle_confidence = 1.0 - shoulder_to_hip
            if triangle_confidence > confidence:
                shape = 'Triangle'
                confidence = triangle_confidence
                
                # Distinguish between Triangle and Spoon
                if waist_to_hip < 0.8:
                    spoon_confidence = triangle_confidence + (1.0 - waist_to_hip) * 0.5
                    if spoon_confidence > confidence:
                        shape = 'Spoon'
                        confidence = spoon_confidence
        
        # Inverted Triangle: shoulders are wider than hips
        # Less common in women naturally, so we need to check for hourglass more carefully
        if (shoulder_to_hip > 1.1):
            # Check if hourglass is possible despite wider shoulders
            if waist_to_hip < 0.75 and shoulder_to_waist > 1.15:
                # This might be an hourglass with slightly wider shoulders
                hourglass_alt_confidence = 0.8 * (1.0 - waist_to_hip) * (shoulder_to_waist - 1.0)
                if hourglass_alt_confidence > confidence:
                    shape = 'Hourglass'
                    confidence = hourglass_alt_confidence
            else:
                # Standard inverted triangle calculation
                inverted_triangle_confidence = shoulder_to_hip - 1.0
                if inverted_triangle_confidence > confidence:
                    shape = 'Inverted Triangle'
                    confidence = inverted_triangle_confidence
        
        # If no shape was determined, default to Rectangle with low confidence
        if shape is None:
            shape = 'Rectangle'
            confidence = 0.3
            
        # Scale confidence to 0-100%
        confidence = min(confidence * 100, 100)
        
        # Print chosen shape and confidence
        print(f"Selected shape: {shape} with confidence {confidence:.1f}%")
        
        # Get description for the determined shape
        description = self.get_shape_description(shape)
        
        return {
            'shape': shape,
            'confidence': confidence,
            'description': description
        }
    
    def get_shape_description(self, shape):
        """
        Get a description of the body shape and typical styling recommendations.
        
        Args:
            shape: Body shape category
            
        Returns:
            Description string
        """
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
                "Your shoulders and hips are balanced with a significantly "
                "smaller waist, creating a curved silhouette. "
                "Styling tip: Highlight your waist with fitted or belted pieces, "
                "and choose clothes that follow your natural curves."
            )
        }
        
        return descriptions.get(shape, "Body shape analysis unavailable.") 