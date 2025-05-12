# Sample Images for Body Shape Classification

This directory contains sample images for testing the body shape classification system.

## Using Sample Images

1. Place full-body images in this directory
2. Run the test script:
   ```
   python test_predict.py --image static/sample_images/your_image.jpg --height 170
   ```
   
   Replace `your_image.jpg` with your image filename and `170` with the height in cm of the person in the image.

## Image Requirements

- Full-body image (head to toe)
- Person facing the camera directly
- Relatively form-fitting clothing for better accuracy
- Clear, uncluttered background (if possible)

## Expected Results

The system will output:
- A visualization with keypoints and measurements
- The predicted body shape with confidence score
- A description of the body shape with styling recommendations

For actual predictions with real model execution, use `predict.py` instead of the test script. 