import os
import cv2
import matplotlib.pyplot as plt
from body_shape_prediction import predict_body_shape

"""
Sample script demonstrating how to use the body shape prediction in your code.
This shows how to incorporate the prediction as part of a larger application.
"""

def process_images(image_directory, height_cm):
    """Process multiple images and display results."""
    # Get all image files from the directory
    image_files = [f for f in os.listdir(image_directory) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        print(f"\nProcessing: {image_file}")
        
        # Call the prediction function
        result = predict_body_shape(
            image_path=image_path,
            height_cm=height_cm,
            output_path=os.path.join('results', f"result_{image_file}"),
            debug=False
        )
        
        if result:
            results.append({
                'image': image_file,
                'shape': result['shape'],
                'confidence': result['confidence'],
                'features': result['features']
            })
    
    return results

def analyze_results(results):
    """Analyze and display results from multiple predictions."""
    # Count occurrences of each body shape
    shape_counts = {}
    for result in results:
        shape = result['shape']
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    # Display body shape distribution
    print("\nBody Shape Distribution:")
    for shape, count in shape_counts.items():
        print(f"{shape}: {count} images ({count/len(results)*100:.1f}%)")
    
    # Calculate average measurements
    avg_features = {}
    for feature_name in results[0]['features'].keys():
        values = [r['features'][feature_name] for r in results if r['features'][feature_name] is not None]
        if values:
            avg_features[feature_name] = sum(values) / len(values)
    
    print("\nAverage Measurements:")
    for feature, value in avg_features.items():
        if 'cm' in feature:
            print(f"{feature}: {value:.1f} cm")
        elif 'ratio' in feature:
            print(f"{feature}: {value:.2f}")
    
    # Create a simple visualization
    plt.figure(figsize=(10, 6))
    plt.bar(shape_counts.keys(), shape_counts.values())
    plt.title('Body Shape Distribution')
    plt.xlabel('Body Shape')
    plt.ylabel('Count')
    plt.savefig('body_shape_distribution.png')
    print("\nSaved distribution chart to body_shape_distribution.png")

def main():
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Process images from sample directory - change this to your image directory
    results = process_images('static/sample_images', height_cm=170)
    
    # Only analyze if we have results
    if results:
        analyze_results(results)
    else:
        print("No valid results to analyze.")

if __name__ == "__main__":
    main() 