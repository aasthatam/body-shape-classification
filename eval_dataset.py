import os
import csv
import pandas as pd
from body_shape_prediction import predict_body_shape

def load_measurements():
    """Load ground truth measurements from CSV file."""
    measurements_path = 'evaluation_dataset/measurements.csv'
    return pd.read_csv(measurements_path)

def evaluate_dataset():
    """Process all images in the evaluation dataset and compare results."""
    # Load measurements data
    measurements_df = load_measurements()
    
    # Create output directory if it doesn't exist
    output_dir = 'evaluation_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create results CSV file
    results_file = os.path.join(output_dir, 'results.csv')
    
    # Open results file for writing
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'height_cm', 
                     'front_bust', 'front_waist', 'front_hips',  # Ground truth
                     'measured_shoulder', 'measured_waist', 'measured_hip',  # Measured values
                     'shoulder_hip_ratio', 'waist_hip_ratio',  # Key ratios
                     'shape', 'confidence', 'is_female']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each image
        for _, row in measurements_df.iterrows():
            filename = row['filename']
            height_cm = row['height_cm']
            
            image_path = os.path.join('evaluation_dataset', 'Images_Blurred', filename)
            output_path = os.path.join(output_dir, f"female_{filename}")
            
            print(f"\nProcessing {filename} with height {height_cm}cm...")
            
            # Process with female hip adjustment (default)
            result = predict_body_shape(
                image_path=image_path,
                height_cm=height_cm,
                output_path=output_path,
                debug=True,
                is_female=True
            )
            
            if result:
                features = result['features']
                
                # Write results to CSV
                writer.writerow({
                    'filename': filename,
                    'height_cm': height_cm,
                    'front_bust': row['front_bust'],
                    'front_waist': row['front_waist'],
                    'front_hips': row['front_hips'],
                    'measured_shoulder': round(features['shoulder_width_cm'], 1),
                    'measured_waist': round(features['waist_width_cm'], 1),
                    'measured_hip': round(features['hip_width_cm'], 1),
                    'shoulder_hip_ratio': round(features['shoulder_to_hip_ratio'], 2),
                    'waist_hip_ratio': round(features['waist_to_hip_ratio'], 2),
                    'shape': result['shape'],
                    'confidence': round(result['confidence'], 1),
                    'is_female': True
                })
                
                print(f"  Results for {filename}:")
                print(f"  - Shape: {result['shape']} (Confidence: {result['confidence']:.1f}%)")
                print(f"  - Shoulder-to-Hip Ratio: {features['shoulder_to_hip_ratio']:.2f}")
                print(f"  - Waist-to-Hip Ratio: {features['waist_to_hip_ratio']:.2f}")
                print(f"  - Hip Width: {features['hip_width_cm']:.1f}cm")
            else:
                print(f"  Failed to process {filename}")
        
    print(f"\nEvaluation complete. Results saved to {results_file}")

if __name__ == "__main__":
    evaluate_dataset() 