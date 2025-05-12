# Body Shape Classification System

A computer vision system that classifies human body shape from a single 2D image and height information.

## Overview

This system implements a pipeline consisting of four main steps:
1. **Instance Segmentation** - Removing background from input images
2. **Key Point Detection** - Identifying body joints
3. **Hip Keypoint Correction** - Using silhouette to improve hip width measurement
4. **Geometric Feature Extraction** - Calculating body proportions
5. **Body Shape Classification** - Classifying into one of five categories:
   - Rectangle
   - Triangle
   - Inverted Triangle
   - Spoon
   - Hourglass

## Setup

1. Clone this repository
2. Install setuptools first to avoid distutils errors:
   ```
   pip install setuptools
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download sample images for testing:
   ```
   python download_samples.py
   ```
5. Run the system check to verify your installation:
   ```
   python check_system.py
   ```
6. Run the prediction:
   ```
   python body_shape_prediction.py --image static/sample_images/sample_person.jpg --height height_in_cm
   ```

## Usage Options

### Option 1: Quick Prediction

For straightforward body shape prediction:

```
python body_shape_prediction.py --image path/to/image.jpg --height 170 --output results.jpg
```

Add the `--debug` flag for more detailed output:

```
python body_shape_prediction.py --image path/to/image.jpg --height 170 --debug
```

To skip the silhouette-based hip correction (use original MediaPipe hip points only):

```
python body_shape_prediction.py --image path/to/image.jpg --height 170 --skip-correction
```

### Option 2: Test Hip Keypoint Correction

To test and visualize the hip keypoint correction specifically:

```
python test_hip_correction.py --image path/to/image.jpg
```

This will generate several debug images showing the original keypoints, corrected keypoints,
and a side-by-side comparison.

### Option 2: Prepare Your Image

If you want to prepare your image first:

```
python prepare_image.py --image path/to/raw_image.jpg
python body_shape_prediction.py --image prepared_raw_image.jpg --height 170
```

### Option 3: Test with Simulation

If you don't have the required models or want to test without downloading large model files:

```
python test_predict.py --image path/to/image.jpg --height 170
```
Note: This uses a simulation and doesn't provide actual predictions.

## Troubleshooting

### Common Issues

#### "Failed to load image"

If you're seeing image loading errors, use the image test tool:

```
python image_test.py --image path/to/your/image.jpg
```

This will diagnose common image issues such as incorrect file paths, unsupported formats, or file permissions problems.

#### No Sample Images

If you need sample images to test the system:

```
python download_samples.py
```

This will download sample images known to work with the system.

#### General Troubleshooting

For general system diagnostics:

```
python check_system.py
```

This will check your environment, dependencies, and file structure.

For more detailed troubleshooting, refer to the SETUP_GUIDE.md file.

## Image Requirements

For best results:
- Full-body image (head to toe)
- Person facing the camera directly
- Relatively form-fitting clothing for better accuracy
- Clear, uncluttered background (if possible)

## Project Structure

- `app/` - Main application code
  - `models/` - ML models for segmentation, pose estimation, and classification
  - `utils/` - Utility functions for image processing and feature extraction
- `data/` - Sample data and model weights
- `static/` - Sample images and results
- `body_shape_prediction.py` - Main prediction script
- `prepare_image.py` - Image preparation utility
- `test_predict.py` - Test prediction with simulation
- `check_system.py` - System diagnostics tool
- `image_test.py` - Image loading diagnostics tool
- `download_samples.py` - Sample image downloader

## Features

### Hip Keypoint Correction

A common issue with pose estimation models is that hip keypoints are often detected incorrectly,
especially when:
- Arms are covering the hips
- The person is wearing loose clothing
- The pose is non-neutral

Our system addresses this issue by:
1. Using MediaPipe to get initial hip keypoint estimates
2. Extracting the body silhouette from segmentation
3. Finding the actual body boundaries at the hip level
4. Correcting the hip keypoints based on these boundaries

This significantly improves the accuracy of hip width measurements, which is crucial for 
distinguishing between body shapes like hourglass, rectangle, and inverted triangle. 